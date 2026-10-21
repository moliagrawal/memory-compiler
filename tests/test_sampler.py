"""Unit tests for video frame sampling."""

import numpy as np
from mc.data.sampler import VideoSampler


def test_synthetic_sampler_deterministic():
    sampler = VideoSampler(fps=1.0)
    frames1, ts1, meta1 = sampler.generate_synthetic_video_frames(duration=10.0, fps=1.0)
    frames2, ts2, meta2 = sampler.generate_synthetic_video_frames(duration=10.0, fps=1.0)

    assert frames1.shape == (10, 224, 224, 3)
    assert len(ts1) == 10
    assert np.array_equal(frames1, frames2)
    assert np.array_equal(ts1, ts2)
    assert meta1["sampled_fps"] == 1.0


def test_sampler_fps_rates():
    for target_fps in [0.5, 1.0, 2.0]:
        frames, ts, meta = VideoSampler.generate_synthetic_video_frames(duration=20.0, fps=target_fps)
        expected_frames = int(20.0 * target_fps)
        assert frames.shape[0] == expected_frames
        assert len(ts) == expected_frames
