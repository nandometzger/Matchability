# Matchability sensitivity study

- pairs: 5  ·  videos: 0001, 0002, 0003, 0004, 0005
- `R_pred = distort(R_gt)` for each severity; DeDoDe v2 on MPS, 768 px, 5000 kpts.

| distortion | expected | E_match min → max | SSIM min → max | PSNR(dB) min → max |
| --- | --- | --- | --- | --- |
| disparity_scale | flat | 20.1% → 43.5% | 0.90 → 0.70 | 30.1 → 18.6 |
| horizontal_shift | flat | 12.1% → 27.5% | 0.82 → 0.66 | 27.9 → 16.8 |
| contrast_fade | rises | 7.8% → 83.8% | 1.00 → 0.76 | 34.7 → 15.7 |
| downscale_upscale | rises | 27.8% → 95.6% | 0.97 → 0.81 | 39.5 → 28.5 |
| elastic_warp | rises | 22.9% → 90.8% | 0.96 → 0.71 | 35.8 → 21.1 |
| gaussian_blur | rises | 0.0% → 99.7% | 1.00 → 0.79 | 100.0 → 25.0 |
| gaussian_noise | rises | 16.6% → 94.6% | 0.98 → 0.07 | 41.8 → 11.6 |
| jpeg | rises | 16.4% → 80.1% | 0.99 → 0.81 | 44.7 → 26.7 |
| occlusion_patch | rises | 8.3% → 80.4% | 0.99 → 0.78 | 32.1 → 15.9 |
| vertical_shift | rises | 24.7% → 97.6% | 0.96 → 0.73 | 38.0 → 23.9 |
