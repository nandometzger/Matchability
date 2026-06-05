# Matchability sensitivity study

- backend: `dedode`  ·  pairs: 5  ·  resolution: 512px  ·  keypoints: 5000  ·  tau: 2.0px
- `E_match` averaged over all pairs, reported as a percentage.

| distortion | expected | E_match @ min sev | E_match @ max sev | SSIM min→max |
| --- | --- | --- | --- | --- |
| brightness_gamma | flat | 5.5% (sev 1.0) | 37.1% (sev 3.0) | 1.00→0.29 |
| contrast_fade | rises | 0.0% (sev 1.0) | 83.8% (sev 0.1) | 1.00→0.76 |
| disparity_scale | flat | 0.0% (sev 1.0) | 43.5% (sev 1.3) | 1.00→0.70 |
| downscale_upscale | rises | 0.0% (sev 1.0) | 95.6% (sev 0.1) | 1.00→0.81 |
| elastic_warp | rises | 0.0% (sev 0.0) | 90.8% (sev 16.0) | 1.00→0.71 |
| gaussian_blur | rises | 0.0% (sev 0.0) | 99.7% (sev 16.0) | 1.00→0.79 |
| gaussian_noise | rises | 0.0% (sev 0.0) | 94.6% (sev 80.0) | 1.00→0.07 |
| horizontal_shift | flat | 0.0% (sev 0.0) | 27.5% (sev 64.0) | 1.00→0.66 |
| identity | anchor_low | 0.0% (sev 0.0) | 0.0% (sev 0.0) | 1.00→1.00 |
| jpeg | rises | 16.4% (sev 95.0) | 80.1% (sev 5.0) | 0.99→0.81 |
| occlusion_patch | rises | 1.8% (sev 0.0) | 80.4% (sev 0.6) | 1.00→0.78 |
| scramble | anchor_high | 96.2% (sev 1.0) | 96.2% (sev 1.0) | 0.50→0.50 |
| vertical_shift | rises | 0.0% (sev 0.0) | 97.6% (sev 8.0) | 1.00→0.73 |
