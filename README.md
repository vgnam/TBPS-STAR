# TBPS test-time adaptation experiments

Experiments on test-time adaptation (TTA) for Text-Based Person Search (TBPS) in the Pretrain-then-Adapt setting: a CLIP/IRRA model pretrained on synthetic data (the MLLM4Text-ReID checkpoint `best0.pth`) is adapted to an unlabeled target test split.

The main method is **STAR** (balanced optimal-transport pseudo-labels for adaptation). It is compared against UATTA (SIGIR 2026), TCR (ICLR 2025), Tent, SAR and READ on the same checkpoint.

## Results (R1 / mAP, MLLM4Text-ReID checkpoint)

| Method | RSTPReid | CUHK-PEDES | ICFG-PEDES |
|---|---|---|---|
| Zero-shot | 49.00 / 37.25 | 57.62 / 51.45 | 37.27 / 20.14 |
| UATTA | 52.00 / 39.08 | 58.12 / 52.06 | 40.55 / 21.65 |
| TCR | 53.45 / 39.82 | 59.97 / 53.94 | 42.46 / 24.33 |
| Sinkhorn rerank only (no training) | 54.50 / 41.49 | 61.70 / 55.10 | 42.45 / 25.13 |
| **STAR, cosine** | **56.65 / 42.27** | **62.16 / 55.84** | **45.49 / 26.03** |
| **STAR + Sinkhorn** | **57.03 / 43.27** | **64.83 / 57.92** | **47.18 / 27.69** |

Notes on the table:

- **UATTA epoch selection:** on RSTP and CUHK, UATTA reports its best epoch chosen on the test set. On ICFG its best epoch is also its last.
- **STAR protocol:** STAR is reported at a fixed epoch 10. RSTP and CUHK are the mean of 3 seeds (std ≤ 0.2 R1); ICFG uses one seed.
- **Ablation:** with greedy-argmax pseudo-labels instead of OT, the same recipe collapses on ICFG (cosine R1 26.08).

## STAR

1. **Balanced assignment.** Encode all target captions and compute similarities S to the frozen gallery. Run log-domain Sinkhorn with uniform marginals (ε = 0.05) to get a hub-free transport plan P.
2. **Adaptation (EM).** Use the hard OT labels of the 60% most confident captions as targets for a cross-entropy at τ = 0.05. Only the text-transformer LayerNorms are updated (AdamW, lr 3e-4, wd 1e-4, batch 64). Recompute P every epoch.
3. **Inference.** Use either plain cosine with the adapted encoder, or Sinkhorn reranking on the adapted scores.

## Layout

| Path | Contents |
|---|---|
| `kernel_star/` | STAR adaptation (`star_adapt.py`) and its Kaggle kernel (RSTP, CUHK) |
| `kernel_icfg/` | ICFG-PEDES kernel: STAR (OT vs greedy labels) and TCR |
| `kernel_baselines/` | Tent / SAR / READ / TCR, ported from the TCR release (`tta_baselines.py`) |
| `kernel_uatta/`, `kernel_uatta_icfg/` | UATTA reproduction on the MLLM4Text-ReID checkpoint (ICFG kernel includes memory patches) |
| `ratbps/` | Feature-level experiments: training-free STAR steps (`star.py`), subspace probes (`subspace.py`), RA-TBPS on CLIP (`clip_tta.py`) |
| `opentbps/` | Earlier open-set TBPS experiments (calibration, ITM verification) |
| `kernel/`, `kernel_aptm/`, `kernel_itm/`, `kernel_ratbps/` | Earlier Kaggle jobs (IRRA/APTM embeddings, BLIP/APTM ITM, RA-TBPS) |
| `results/` | Logs and JSON results of all runs (feature `.npz` files and weights are not included) |

## Running

Every `kernel_*/build.py` writes a self-contained Kaggle script to `kernel_*/out/`. Push it with:

```bash
python kernel_star/build.py && kaggle kernels push -p kernel_star/out
```

The kernels clone [UATTA](https://github.com/nkuzjh/UATTA) (IRRA_UATTA code) and download everything else themselves:

- the MLLM4Text-ReID checkpoint from Hugging Face (`TwT-6/MLLM4Text-ReID`);
- CUHK-PEDES and ICFG-PEDES from Hugging Face (`cjc/CHUK-PEDES`, `cjc/ICFG-PEDES`);
- RSTPReid from its Google Drive link.

Datasets, checkpoints, embeddings and third-party repos (IRRA, APTM) are not part of this repository.
