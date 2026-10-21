#!/usr/bin/env python3
"""CLI Interactive Demo Walkthrough for Memory Compiler."""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def run_demo():
    print("=" * 65)
    print("⚡ MEMORY COMPILER - LIVE SYSTEM DEMONSTRATION")
    print("=" * 65)

    # 1. System Health Check
    health = requests.get(f"{BASE_URL}/health").json()
    print(f"\n[1] Hardware & VRAM Status:")
    print(f"    • GPU Device:      {health['gpu']['device']}")
    print(f"    • Peak VRAM:       {health['gpu']['peak_vram_mb']} MB / 4,096 MB")
    print(f"    • Safe Under 3.5G: {health['vram_limit_safe']}")

    # 2. List Videos
    videos = requests.get(f"{BASE_URL}/api/videos").json()
    print(f"\n[2] Available Demo Videos ({len(videos['videos'])} registered):")
    for v in videos["videos"][:3]:
        print(f"    • {v['filename']} (Indexed: {v['indexed']})")

    # 3. Grounding Scenarios
    scenarios = [
        ("TR_0001", "person opens the refrigerator door"),
        ("demo_kitchen_routine", "taking glass and washing at sink"),
        ("demo_living_room", "first stands up then walks to window")
    ]

    for vid, q in scenarios:
        print(f"\n[3] Grounding Query on {vid}.mp4:")
        print(f"    Query: \"{q}\"")
        t0 = time.time()
        res = requests.post(f"{BASE_URL}/api/search", json={
            "query": q,
            "video_id": vid,
            "model_mode": "COMPARE",
            "top_k": 3
        }).json()
        latency = res.get("latency_ms", round((time.time() - t0) * 1000, 2))

        print(f"    Latency: {latency} ms")
        print(f"    Top Grounded Moments:")
        for pred in res["primary_predictions"]:
            print(f"      Rank #{pred['rank']}: [{pred['start']:4.1f}s – {pred['end']:4.1f}s]  Conf: {pred['score']*100:5.1f}%  (Relevance: {pred['rel_score']*100:.1f}%, Boundary: {pred['boundary_score']*100:.1f}%)")

        print(f"    Live Comparison:")
        b1_top = res["comparison"]["B1_frame_similarity"][0]
        full_top = res["comparison"]["FULL_system"][0]
        print(f"      • B1 (Frame Similarity): [{b1_top['start']}s – {b1_top['end']}s] (Fixed window width)")
        print(f"      • Full System (B3+Rerank): [{full_top['start']}s – {full_top['end']}s] (Learned boundary heads)")

    print("\n" + "=" * 65)
    print("Demo complete! View the interactive visual studio at: http://localhost:8000")
    print("=" * 65)

if __name__ == "__main__":
    run_demo()
