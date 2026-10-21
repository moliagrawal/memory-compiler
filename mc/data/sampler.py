"""Video sampling module supporting fixed-fps frame extraction with decord/OpenCV."""

import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class VideoSampler:
    """
    Fixed-FPS video sampler.
    Extracts frames deterministically at specified fps (e.g. 0.5, 1.0, 2.0 fps)
    and returns numpy array of RGB frames and array of corresponding timestamps in seconds.
    """

    def __init__(self, fps: float = 1.0, max_frames: Optional[int] = 512):
        self.fps = float(fps)
        self.max_frames = max_frames

    def sample_video(self, video_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Sample frames and timestamps from video file.
        Returns:
            frames: np.ndarray of shape [T, H, W, 3], uint8 RGB
            timestamps: np.ndarray of shape [T], float32 (seconds)
            metadata: dict with original fps, total_frames, duration
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Try decord first for high efficiency
        try:
            import decord
            decord.bridge.set_bridge("native")
            vr = decord.VideoReader(video_path, num_threads=1)
            total_frames = len(vr)
            native_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() > 0 else 24.0
            duration = total_frames / native_fps

            # Calculate target timestamps
            interval = 1.0 / self.fps
            target_timestamps = np.arange(0.0, duration, interval, dtype=np.float32)
            if self.max_frames and len(target_timestamps) > self.max_frames:
                target_timestamps = target_timestamps[:self.max_frames]

            frame_indices = (target_timestamps * native_fps).astype(np.int64)
            frame_indices = np.clip(frame_indices, 0, total_frames - 1)

            # Decord batch extraction
            frames_batch = vr.get_batch(frame_indices.tolist()).asnumpy()
            meta = {
                "native_fps": native_fps,
                "total_frames": total_frames,
                "duration": duration,
                "sampled_fps": self.fps,
                "num_sampled": len(frames_batch)
            }
            return frames_batch, target_timestamps, meta

        except ImportError:
            # Fallback to OpenCV
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video with OpenCV: {video_path}")

            native_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / native_fps

            interval = 1.0 / self.fps
            target_timestamps = np.arange(0.0, duration, interval, dtype=np.float32)
            if self.max_frames and len(target_timestamps) > self.max_frames:
                target_timestamps = target_timestamps[:self.max_frames]

            frame_indices = set((target_timestamps * native_fps).astype(np.int64))

            frames = []
            timestamps_collected = []
            curr_idx = 0

            while cap.isOpened() and (self.max_frames is None or len(frames) < self.max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                if curr_idx in frame_indices:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    timestamps_collected.append(curr_idx / native_fps)
                curr_idx += 1

            cap.release()
            frames_np = np.array(frames, dtype=np.uint8) if frames else np.zeros((0, 224, 224, 3), dtype=np.uint8)
            ts_np = np.array(timestamps_collected, dtype=np.float32)

            meta = {
                "native_fps": native_fps,
                "total_frames": total_frames,
                "duration": duration,
                "sampled_fps": self.fps,
                "num_sampled": len(frames_np)
            }
            return frames_np, ts_np, meta

    @staticmethod
    def generate_synthetic_video_frames(
        duration: float = 30.0,
        fps: float = 1.0,
        height: int = 224,
        width: int = 224
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate deterministic synthetic video frames for tests, fast benchmarking,
        and continuous integration without large video files.
        """
        num_frames = int(duration * fps)
        timestamps = np.arange(num_frames, dtype=np.float32) / fps

        # Deterministic visual pattern based on frame index
        frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
        for t in range(num_frames):
            color = ((t * 17) % 256, (t * 43) % 256, (t * 79) % 256)
            frames[t, :, :] = color

        meta = {
            "native_fps": 30.0,
            "total_frames": int(duration * 30.0),
            "duration": duration,
            "sampled_fps": fps,
            "num_sampled": num_frames
        }
        return frames, timestamps, meta
