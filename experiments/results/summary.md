# Matchability sensitivity study

- pairs: 5  ·  videos: 0001, 0002, 0003, 0004, 0005
- `R_pred = distort(R_gt)` for each severity; DeDoDe v2 on MPS, 768 px, 5000 kpts.

| distortion | expected | E_match min → max | SSIM min → max | PSNR(dB) min → max |
| --- | --- | --- | --- | --- |
| brightness_gamma | flat | 5.5% → 37.1% | 1.00 → 0.29 | 57.0 → 10.7 |
| contrast_fade | rises | 0.0% → 83.8% | 1.00 → 0.76 | 100.0 → 15.7 |
| disparity_scale | flat | 0.0% → 43.5% | 1.00 → 0.70 | 100.0 → 18.6 |
| downscale_upscale | rises | 0.0% → 95.6% | 1.00 → 0.81 | 100.0 → 28.5 |
| elastic_warp | rises | 0.0% → 90.8% | 1.00 → 0.71 | 100.0 → 21.1 |
| gaussian_blur | rises | 0.0% → 99.7% | 1.00 → 0.79 | 100.0 → 25.0 |
| gaussian_noise | rises | 0.0% → 94.6% | 1.00 → 0.07 | 100.0 → 11.6 |
| horizontal_shift | flat | 0.0% → 27.5% | 1.00 → 0.66 | 100.0 → 16.8 |
| jpeg | rises | 16.4% → 80.1% | 0.99 → 0.81 | 44.7 → 26.7 |
| occlusion_patch | rises | 1.8% → 80.4% | 1.00 → 0.78 | 69.3 → 15.9 |
| vertical_shift | rises | 0.0% → 97.6% | 1.00 → 0.73 | 100.0 → 23.9 |
