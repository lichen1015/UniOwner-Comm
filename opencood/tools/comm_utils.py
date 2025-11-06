
class CommRecorder:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CommRecorder, cls).__new__(cls, *args, **kwargs)
            cls._instance.frame_counter = 0
            cls._instance.volume = 0.0
            cls._instance.pose_bytes = 6.0   # 注意：这是“字节数”，不要写6.0表示“6个数”
            cls._instance.idx = 0
        return cls._instance

    # ===== 新增：一键清零 =====
    def reset(self):
        self.frame_counter = 0
        self.volume = 0.0
        self.idx = 0

    def set_idx(self, idx): self.idx = idx
    def set_pose_bytes(self, bytes): self.pose_bytes = float(bytes)

    def add_feature_map(self, C, H, W, nums=1, ratio=1.0, bytes_per_element=4):
        self.volume += float(C) * float(H) * float(W) * float(bytes_per_element) * float(ratio) * float(nums)

    def add_pose_bytes(self, nums=1):
        self.volume += float(self.pose_bytes) * float(nums)

    def add_direct(self, bytes):
        self.volume += float(bytes)

    def increase_frame_counter(self, num=1):
        self.frame_counter += int(num)

    def get_frame_counter(self):
        return self.frame_counter

    def get_idx(self):
        return self.idx

    def get_avg_comm(self):
        return (self.volume / self.frame_counter) if self.frame_counter > 0 else 0.0

    def get_format_bandwidth(self, frame_rate=10):
        if self.frame_counter == 0:
            return "0 b/s"
        bandwidth = self.volume / self.frame_counter * frame_rate * 8  # bits/s
        if bandwidth < 1024:
            return f"{bandwidth:.3f} b/s"
        elif bandwidth < 1024 * 1024:
            return f"{bandwidth / 1024:.3f} Kb/s"
        else:
            return f"{bandwidth / 1024 / 1024:.3f} Mb/s"

comm_recorder = CommRecorder()
