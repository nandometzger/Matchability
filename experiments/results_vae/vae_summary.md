# Matchability sensitivity study

| VAE | repo | size |
| --- | --- | --- |
| TAESD | `madebyollin/taesd` | AutoencoderTiny/KL |
| TAESDXL | `madebyollin/taesdxl` | AutoencoderTiny/KL |
| TAESD3 | `madebyollin/taesd3` | AutoencoderTiny/KL |
| TAEF1 | `madebyollin/taef1` | AutoencoderTiny/KL |
| SD-VAE-MSE | `stabilityai/sd-vae-ft-mse` | AutoencoderTiny/KL |
| SD-VAE-EMA | `stabilityai/sd-vae-ft-ema` | AutoencoderTiny/KL |
| SDXL-VAE | `madebyollin/sdxl-vae-fp16-fix` | AutoencoderTiny/KL |
| DC-AE | `mit-han-lab/dc-ae-f16c16-sana-1.1` | AutoencoderTiny/KL |

- backend: `dedode`  ·  pairs: 5  ·  resolution: 768px  ·  keypoints: 5000  ·  tau: 2.0px

| distortion | expected | E_match min → max | SSIM min → max | PSNR(dB) min → max |
| --- | --- | --- | --- | --- |
| TAESD | rises | 50.1% → 50.1% | 0.89 → 0.89 | 25.6 → 25.6 |
| TAESDXL | rises | 49.8% → 49.8% | 0.90 → 0.90 | 32.4 → 32.4 |
| TAESD3 | rises | 36.2% → 36.2% | 0.95 → 0.95 | 31.3 → 31.3 |
| TAEF1 | rises | 37.5% → 37.5% | 0.96 → 0.96 | 33.7 → 33.7 |
| SD-VAE-MSE | rises | 41.9% → 41.9% | 0.92 → 0.92 | 34.4 → 34.4 |
| SD-VAE-EMA | rises | 41.8% → 41.8% | 0.92 → 0.92 | 34.0 → 34.0 |
| SDXL-VAE | rises | 38.7% → 38.7% | 0.93 → 0.93 | 34.8 → 34.8 |
