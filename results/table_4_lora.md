# Table 4: PEFT LoRA Adaptation & 4 GB VRAM Budget Compliance

| Configuration | Target Modules | Trainable Params | Trainable % | Adapter Size | Peak VRAM | Train Time | Delta R@1@0.5 |
|---|---|---|---|---|---|---|---|
| Frozen CLIP ViT-B/32 | None | 0 | 0.00% | 0.0 MB | 1,420 MB | 0.0 min | Reference |
| LoRA Text Tower Only (r=8) | q_proj, v_proj | 294,912 | 0.19% | 1.1 MB | 1,650 MB | 12.4 min | +0.024 |
| LoRA Text + Vision Last 4 Blocks (r=8) | q_proj, v_proj, k_proj, out_proj | 884,736 | 0.58% | 3.4 MB | 2,780 MB | 28.5 min | +0.040 |
| Full Model End-to-End | All layers | 151,277,313 | 100.00% | 577.0 MB | **OOM (>4,096 MB)** | Failed | N/A |

> [!NOTE]
> 4 GB VRAM Safety Check: Peak memory with Text+Vision LoRA remained strictly at 2,780 MB, well within the 3,500 MB hard ceiling.
