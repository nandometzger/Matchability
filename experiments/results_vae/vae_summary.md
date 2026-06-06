# Matchability sensitivity study

| VAE | repo | class |
| --- | --- | --- |
| TAESD | `madebyollin/taesd` | AutoencoderTiny |
| TAESDXL | `madebyollin/taesdxl` | AutoencoderTiny |
| TAESD3 | `madebyollin/taesd3` | AutoencoderTiny |
| TAEF1 | `madebyollin/taef1` | AutoencoderTiny |
| SD-VAE-MSE | `stabilityai/sd-vae-ft-mse` | AutoencoderKL |
| SD-VAE-EMA | `stabilityai/sd-vae-ft-ema` | AutoencoderKL |
| SDXL-VAE | `madebyollin/sdxl-vae-fp16-fix` | AutoencoderKL |

- backend: `dedode`  ·  pairs: 5  ·  resolution: 768px  ·  keypoints: 5000  ·  tau: 2.0px

| VAE | E_match | SSIM | PSNR(dB) |
| --- | --- | --- | --- |
| TAESD | 50.1% | 0.89 | 25.6 |
| TAESDXL | 49.8% | 0.90 | 32.4 |
| TAESD3 | 36.2% | 0.95 | 31.3 |
| TAEF1 | 37.5% | 0.96 | 33.7 |
| SD-VAE-MSE | 41.9% | 0.92 | 34.4 |
| SD-VAE-EMA | 41.8% | 0.92 | 34.0 |
| SDXL-VAE | 38.7% | 0.93 | 34.8 |
