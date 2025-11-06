#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修改 YAML 中的 where2comm.selector.{mode, topk} 并启动分布式训练

用法示例（与你给的命令等价）：
python modify_and_train.py \
  -y opencood/hypes_yaml/dairv2x/lidar_only/pointpillar_selective_comm.yaml \
  --mode heuristic --topk 4 \
  --project vis_noise_byadd/heuristic_k4_bs4 \
  --gpus 0,1,2,3 \
  --nproc 4 \
  --workdir .

说明：
- 默认会将修改后的 YAML 写到同目录下的新文件：
  <原名去掉扩展>.<mode>_k<topk>.yaml
- 若未安装 ruamel.yaml，将回退到 PyYAML（可能丢注释/锚点，但键值会正确）
- 若只想改 YAML 不训练，可加 --no-train
"""

from __future__ import annotations
import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict

def load_yaml_keep_structure(yaml_path: Path) -> Any:
    try:
        from ruamel.yaml import YAML  # 保留注释/锚点/引用
        yaml = YAML(typ='rt')
        yaml.preserve_quotes = True
        with yaml_path.open('r', encoding='utf-8') as f:
            data = yaml.load(f)
        return data, yaml, 'ruamel'
    except Exception:
        # 退回到 PyYAML（不保留注释/锚点）
        import yaml
        with yaml_path.open('r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data, yaml, 'pyyaml'

def dump_yaml(data: Any, yaml_mod: Any, yaml_path: Path, backend: str) -> None:
    if backend == 'ruamel':
        with yaml_path.open('w', encoding='utf-8') as f:
            yaml_mod.dump(data, f)
    else:
        # pyyaml
        with yaml_path.open('w', encoding='utf-8') as f:
            yaml_mod.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def ensure_path(d: Dict, key: str) -> Dict:
    if key not in d or d[key] is None:
        d[key] = {}
    return d[key]

def modify_yaml(yaml_obj: Any, mode: str, topk: int) -> None:
    """
    安全地设置：
    model.args.where2comm.selector.mode = mode
    model.args.where2comm.selector.topk = topk
    如果路径不存在则按层级创建空字典。
    """
    if not isinstance(yaml_obj, dict):
        raise ValueError("YAML 根对象不是字典，无法修改预期字段。")

    model = ensure_path(yaml_obj, 'model')
    args = ensure_path(model, 'args')
    where2comm = ensure_path(args, 'where2comm')
    selector = ensure_path(where2comm, 'selector')

    selector['mode'] = str(mode)
    selector['topk'] = int(topk)

def build_out_yaml_path(in_yaml: Path, mode: str, topk: int) -> Path:
    stem = in_yaml.stem  # 不含扩展名
    suffix = in_yaml.suffix  # .yaml
    return in_yaml.with_name(f"{stem}.{mode}_k{topk}{suffix}")

def run_training(
    out_yaml: Path,
    project: str,
    gpus: str,
    nproc: int,
    workdir: Path,
    extra: list[str] | None = None,
):
    """
    复刻命令：
    CUDA_VISIBLE_DEVICES=<gpus> python -m torch.distributed.launch --nproc_per_node=<nproc> --use_env \
        opencood/tools/train_ddp_syncbn.py -y <out_yaml> -p <project> --run_test [extra...]
    """
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpus

    cmd = [
        sys.executable, "-m", "torch.distributed.launch",
        f"--nproc_per_node={nproc}",
        "--use_env",
        "opencood/tools/train_ddp_syncbn.py",
        "-y", str(out_yaml),
        "-p", project,
        # "--run_test",
    ]
    if extra:
        cmd.extend(extra)

    print("\n[INFO] 工作目录：", workdir.resolve())
    print("[INFO] 即将执行命令：\n  CUDA_VISIBLE_DEVICES={}\n  {}".format(
        env.get('CUDA_VISIBLE_DEVICES', ''), " ".join(cmd)
    ))
    subprocess.run(cmd, cwd=str(workdir), env=env, check=True)

def main():
    parser = argparse.ArgumentParser(description="修改 YAML 后启动分布式训练")
    parser.add_argument("-y", "--yaml", required=True, type=Path, help="原始 YAML 路径")
    parser.add_argument("--mode", required=True, type=str, help="where2comm.selector.mode，新值")
    parser.add_argument("--topk", required=True, type=int, help="where2comm.selector.topk，新值（整数）")
    parser.add_argument("--out", type=Path, default=None, help="输出 YAML 路径（默认自动命名）")
    parser.add_argument("--project", "-p", type=str, default=None,
                        help="训练的 -p 项目名（默认：vis_noise_byadd/<mode>_k<topk>_bs4）")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="CUDA_VISIBLE_DEVICES 列表，默认 0,1,2,3")
    parser.add_argument("--nproc", type=int, default=4, help="每节点进程数 --nproc_per_node，默认 4")
    parser.add_argument("--workdir", type=Path, default=Path("."), help="作为 cwd 运行训练脚本的目录")
    parser.add_argument("--no-train", action="store_true", help="只修改 YAML，不启动训练")
    parser.add_argument("--extra", nargs=argparse.REMAINDER,
                        help="透传给训练脚本的额外参数（-- 后面的都会原样拼接）")
    args = parser.parse_args()

    in_yaml: Path = args.yaml
    if not in_yaml.exists():
        raise FileNotFoundError(f"找不到 YAML：{in_yaml}")

    data, yaml_mod, backend = load_yaml_keep_structure(in_yaml)
    print(f"[INFO] YAML 解析后端：{backend}")

    modify_yaml(data, args.mode, args.topk)

    out_yaml = args.out or build_out_yaml_path(in_yaml, args.mode, args.topk)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    dump_yaml(data, yaml_mod, out_yaml, backend)
    print(f"[OK] 已写出修改后的 YAML：{out_yaml}")

    if args.no_train:
        print("[INFO] 按 --no-train 要求，不启动训练。")
        return

    project = args.project or f"vis_noise_byadd/{args.mode}_k{args.topk}_bs4"
    extra = None
    if args.extra:
        # 删除可能的分隔符 "--"
        extra = [x for x in args.extra if x != "--"]

    run_training(
        out_yaml=out_yaml,
        project=project,
        gpus=args.gpus,
        nproc=args.nproc,
        workdir=args.workdir,
        extra=extra,
    )

if __name__ == "__main__":
    main()
