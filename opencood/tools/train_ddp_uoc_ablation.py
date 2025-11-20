import argparse
import os
import statistics
import glob
import math
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import multi_gpu_utils
from icecream import ic
import tqdm
import shlex
import subprocess
import torch.distributed as dist

def _ddp_sum(val, device):
    """把标量在所有进程上求和并返回 python 数。非分布式时直接返回。"""
    if not dist.is_available() or not dist.is_initialized():
        return float(val)
    t = torch.tensor([float(val)], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())

def _fmt_bytes(n):
    # 人类可读格式
    units = ['B','KB','MB','GB','TB']
    x, i = float(n), 0
    while x >= 1024.0 and i < len(units)-1:
        x /= 1024.0; i += 1
    return f"{x:.2f} {units[i]}"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def run_eval(saved_path, fusion_method, add_noise=False, eval_gpu="0"):
    # 只在 rank0 执行，且确保 DDP 已收尾
    if dist.is_initialized():
        dist.barrier()
        if dist.get_rank() != 0:
            return
        dist.destroy_process_group()

    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    script = "opencood/tools/inference_w_noise.py" if add_noise else "opencood/tools/inference_bandwidth.py"
    cmd = [
        "python", script,
        "--model_dir", saved_path,
        "--fusion_method", fusion_method,
        "--is_vis",
        # "--config", "/abs/path/to/eval.yaml",  # 如有，建议显式传入
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(eval_gpu)
    env["MPLBACKEND"] = "Agg"   # ★ 关键一行：评测强制无交互后端

    print("Running command:", " ".join(shlex.quote(x) for x in cmd),
          f"(CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}, MPLBACKEND={env['MPLBACKEND']})")

    subprocess.run(cmd, env=env, check=True,
                   cwd="/mnt/workspace/collabor/CoSDH-main")  # ★ 设定 cwd，避免相对路径坑


def train_parser():
    parser = argparse.ArgumentParser(description="opencood ddp training")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='yaml config')
    parser.add_argument('--project_name', '-p',default='',
                        help='project_name')
    parser.add_argument('--model_dir', default='',
                        help='resume from checkpoint folder')
    parser.add_argument('--fusion_method', '-f', default="intermediate")
    parser.add_argument("--half", action='store_true',
                        help="train with mixed precision")
    parser.add_argument("--run_test", action='store_true',
                        help="run inference.py")
    parser.add_argument("--no_pattern", action='store_true',
                        help="disable pattern region recognition")
    parser.add_argument("--no_oaohead", action='store_true',
                        help="disable oaohead")
    parser.add_argument("--no_owner", action='store_true',
                        help="disable mono_sender") 
    parser.add_argument('--dist_url', default='env://',
                        help='distributed init method')
    opt = parser.parse_args()
    return opt

def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    hypes['project_name'] = opt.project_name
    multi_gpu_utils.init_distributed_mode(opt)  # sets opt.distributed, opt.world_size, opt.rank, opt.gpu
    # ---- ablation ----
    if opt.no_oaohead:
        print(f"disable use oaohead..")
        hypes['model']['args']['use_oaohead'] = False
    if opt.no_pattern:
        print(f"disable Pattern Region Reogntiion..")
        hypes['model']['args']['where2comm']['communication']['use_pattern'] = False
    if opt.no_owner:
        print(f"disable Pattern Region Reogntiion..")
        hypes['model']['args']['where2comm']['communication']['use_owner'] = False
        
    # ---- seed ----
    seed = hypes.get('train_params', {}).get('seed', 42)
    set_seed(seed + (opt.rank if hasattr(opt, 'rank') else 0))

    # ---- device binding ----
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu if hasattr(opt, 'gpu') else 0)
        device = torch.device(f'cuda:{opt.gpu}' if hasattr(opt, 'gpu') else 'cuda:0')
    else:
        device = torch.device('cpu')

    # ---- effective batch size control ----
    tp = hypes['train_params']
    per_gpu_cfg_bs = int(tp['batch_size'])  # yaml 中的 batch_size（每卡）
    world_size = opt.world_size if getattr(opt, 'distributed', False) else 1
    global_bs_cfg = int(tp.get('global_batch_size', per_gpu_cfg_bs * world_size))

    # 用 grad accumulation 保持有效 batch 恒定
    per_gpu_bs = min(per_gpu_cfg_bs, max(1, global_bs_cfg // world_size))
    accum_steps = int(math.ceil(global_bs_cfg / (per_gpu_bs * world_size)))
    if accum_steps < 1:
        accum_steps = 1

    if opt.rank == 0:
        print(f"[Batch Control] world_size={world_size}, global_batch_size={global_bs_cfg}, "
              f"per_gpu_batch={per_gpu_bs}, accum_steps={accum_steps}")

    if opt.rank == 0:
        print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    # ---- samplers & loaders ----
    num_workers = int(tp.get('num_workers', 8))
    pin_memory = True

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset, shuffle=True, drop_last=True)
        sampler_val   = DistributedSampler(opencood_validate_dataset, shuffle=False, drop_last=False)

        train_loader = DataLoader(
            opencood_train_dataset,
            batch_size=per_gpu_bs,
            sampler=sampler_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=opencood_train_dataset.collate_batch_train,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,
        )
        val_loader = DataLoader(
            opencood_validate_dataset,
            batch_size=int(tp.get('val_batch_size', per_gpu_bs)),  # 显式 val bs
            sampler=sampler_val,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=opencood_train_dataset.collate_batch_train,
            drop_last=False,
            persistent_workers=True if num_workers > 0 else False,
        )
    else:
        train_loader = DataLoader(
            opencood_train_dataset,
            batch_size=per_gpu_bs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=opencood_train_dataset.collate_batch_train,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,
        )
        val_loader = DataLoader(
            opencood_validate_dataset,
            batch_size=int(tp.get('val_batch_size', per_gpu_bs)),
            shuffle=False,  # 验证不打乱
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=opencood_train_dataset.collate_batch_train,
            drop_last=False,
            persistent_workers=True if num_workers > 0 else False,
        )

    if opt.rank == 0:
        print('Creating Model')
    model = train_utils.create_model(hypes)

    # ---- SyncBN for multi-gpu ----
    if opt.distributed and world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)

    # resume / save dir
    lowest_val_loss = float('inf')
    lowest_val_epoch = -1

    # ---- resume or create run dir (DDP-safe) ----
    resume_flag = bool(opt.model_dir)

    if opt.distributed:
        if opt.rank == 0:
            if resume_flag:
                saved_path = opt.model_dir
            else:
                # 仅 rank0 创建目录与写配置
                saved_path = train_utils.setup_train(hypes)
        else:
            saved_path = None

        # 广播路径与是否恢复训练标志
        obj_list = [saved_path, resume_flag]
        dist.broadcast_object_list(obj_list, src=0)
        saved_path, resume_flag = obj_list

        # rank0 确保目录存在，其它 rank 等待
        if opt.rank == 0:
            os.makedirs(saved_path, exist_ok=True)
        dist.barrier()
    else:
        # 单卡：直接 resume 或创建
        saved_path = opt.model_dir if resume_flag else train_utils.setup_train(hypes)
        os.makedirs(saved_path, exist_ok=True)

    # ---- 加载权重 / 初始化 epoch 号 ----
    if resume_flag:
        # 各 rank 读取同一目录的 ckpt（只读，不会写）
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
    else:
        init_epoch = 0
        lowest_val_epoch = -1

    # DDP wrap
    model_without_ddp = model
    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[opt.gpu], output_device=opt.gpu,
            find_unused_parameters=True
        )
        model_without_ddp = model.module

    # ---- loss & optimizer & scheduler ----
    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)

    # 线性缩放 LR 到 global batch：lr *= global_bs / ref_bs
    ref_bs = int(tp.get('ref_batch_size', global_bs_cfg))  # 若没配，就等于当前 global_bs，不改变 lr
    if ref_bs > 0 and global_bs_cfg != ref_bs:
        scale = global_bs_cfg / ref_bs
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] * scale
        if opt.rank == 0:
            print(f"[LR Scaling] ref_bs={ref_bs} → global_bs={global_bs_cfg}, lr scale x{scale:.4f}")

    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)

    # writer only on rank 0
    writer = SummaryWriter(saved_path) if opt.rank == 0 else None

    # half precision training
    scaler = torch.cuda.amp.GradScaler(enabled=opt.half)

    # torchinfo summary only once
    if opt.rank == 0:
        try:
            from torchinfo import summary
            summary(model_without_ddp)
        except Exception:
            pass

    if opt.rank == 0:
        print('Training start')
    epoches = tp['epoches']
    eval_freq = int(tp.get('eval_freq', 1))
    save_freq = int(tp.get('save_freq', 1))
    supervise_single_flag = getattr(opencood_train_dataset, "supervise_single", False)

    global_step = 0
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if opt.distributed:
            sampler_train.set_epoch(epoch)

        model.train()
        if opt.rank == 0:
            for param_group in optimizer.param_groups:
                print('learning rate %f' % param_group["lr"])
            pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)
            
        optimizer.zero_grad(set_to_none=True)
        accum_cnt = 0

        for i, batch_data in enumerate(train_loader):
            if (batch_data is None) or (batch_data['ego']['object_bbx_mask'].sum() == 0):
                # 跳过空 batch，不累计
                continue

            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch

            with torch.cuda.amp.autocast(enabled=opt.half):
                ouput_dict = model(batch_data['ego'])
                final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
                if supervise_single_flag:
                    final_loss = final_loss + criterion(
                        ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single"
                    )

            # 梯度累积：按 accum_steps 归一化
            final_loss = final_loss / accum_steps
            if opt.half:
                scaler.scale(final_loss).backward()
            else:
                final_loss.backward()

            accum_cnt += 1
            if accum_cnt % accum_steps == 0:
                if opt.half:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if opt.rank == 0:
                # 只在 rank-0 记录/刷新进度条
                criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
                if supervise_single_flag:
                    criterion.logging(epoch, i, len(train_loader), writer, suffix="_single", pbar=pbar2)
                pbar2.update(1)

        # epoch end：若最后不足 accum_steps 也执行一次 step（可选）
        if accum_cnt % accum_steps != 0:
            if opt.half:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # ---------- validation with global aggregation ----------
        if (epoch % eval_freq) == 0:
            model.eval()
            loss_sum_local = torch.tensor(0.0, device=device)
            count_local = torch.tensor(0, device=device, dtype=torch.long)

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'])
                    loss_val = criterion(ouput_dict, batch_data['ego']['label_dict']).detach()
                    loss_sum_local += loss_val
                    count_local += 1

            # all_reduce 聚合（按“批次数”平均）
            if opt.distributed:
                dist.all_reduce(loss_sum_local, op=dist.ReduceOp.SUM)
                dist.all_reduce(count_local, op=dist.ReduceOp.SUM)

            valid_ave = (loss_sum_local / count_local.clamp_min(1)).item()

            # 只让 rank 0 打印/写/保存
            if opt.rank == 0:
                print(f'At epoch {epoch}, the validation loss is {valid_ave:.6f}')
                if writer is not None:
                    writer.add_scalar('Validate_Loss', valid_ave, epoch)

                # lowest val loss
                if valid_ave < lowest_val_loss:
                    lowest_val_loss = valid_ave
                    best_path = os.path.join(saved_path, f'net_epoch_bestval_at{epoch+1}.pth')
                    torch.save(model_without_ddp.state_dict(), best_path)
                    # 清理旧 best
                    if lowest_val_epoch != -1:
                        old_best = os.path.join(saved_path, f'net_epoch_bestval_at{lowest_val_epoch}.pth')
                        if os.path.exists(old_best):
                            try:
                                os.remove(old_best)
                            except Exception:
                                pass
                    lowest_val_epoch = epoch + 1

        # save per epoch
        if opt.rank == 0 and (epoch % save_freq == 0):
            torch.save(model_without_ddp.state_dict(),
                       os.path.join(saved_path, f'net_epoch{epoch+1}.pth'))

        scheduler.step()
        # 如果你的 dataset 需要重置内部状态
        opencood_train_dataset.reinitialize()

    if opt.rank == 0:
        print('Training Finished, checkpoints saved to %s' % saved_path)
        
    if opt.run_test:
        run_eval(saved_path, opt.fusion_method)

if __name__ == '__main__':
    main()
