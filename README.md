# ComfyUI-LuxDiT

ComfyUI nodes for [NVIDIA LuxDiT](https://github.com/nv-tlabs/LuxDiT) (lighting estimation with a video
diffusion transformer): image or video frames in, HDR environment map (`.exr`) out.

## Nodes

| Node | What it does |
| --- | --- |
| **LuxDiT Model Loader** | Loads `luxdit_image` or `luxdit_video` (+ optional real-scene LoRA) and the CogVideoX-5b-I2V VAE/scheduler. Downloads weights on first use. |
| **LuxDiT Sampler** | Takes an `IMAGE` (one image, or a batch of video frames) and returns the two 128x256 panoramas the model predicts: tonemapped LDR and log-intensity. |
| **LuxDiT HDR Merge** | Runs NVIDIA's small HDR-merge MLP on the two panoramas → linear HDR `IMAGE` (unclamped), plus a tonemapped preview. |
| **LuxDiT Save EXR** | Writes the HDR map(s) as float `.exr` to the ComfyUI output folder. |

Typical graph: `Load Image → LuxDiT Sampler (+ Model Loader) → LuxDiT HDR Merge → LuxDiT Save EXR`.
A ready-made API-format graph is in [`workflows/luxdit_image_api.json`](workflows/luxdit_image_api.json).

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this repo> ComfyUI-LuxDiT
python ComfyUI-LuxDiT/install.py     # with ComfyUI's python
```

`install.py` installs the requirements and clones the upstream LuxDiT source (pinned commit) into
`third_party/`. The upstream code is **not** included in this repo.

Weights download on first run into `ComfyUI/models/luxdit/`:
`nvidia/LuxDiT` (only the variant you pick + `hdr_merge_mlp`, ~12 GB per variant) and the
`vae/` + `scheduler/` folders of `zai-org/CogVideoX-5b-I2V` (~0.9 GB).

## Photo → 4K HDRI (v2, recommended)

`workflows/photo_to_hdri_4k_v2_api.json` is the best configuration from the 10-image Poly Haven benchmark:
Qwen-Image 2.1 + 360 LoRA (25 steps, CFG 1) with **LuxDiT's lighting map as a second reference image**
(bilinear upscale to 1024×512), **pole repair** (Up/Down cube faces inpainted with plain per-pole prompts
and a negative prompt against invented objects, then pasted back), RealESRGAN 4K, DiffHDR, gain-map merge
(full-colour clip fill), and the **gated sun** (`LuxDiT Sun (gated)`: adds LuxDiT's sun only when it is above
the horizon, agrees with the panorama glow within 30°, and carries ≥15% of the light). Needs the
Qwen-Image-2.1-Panorama, DiffHDR, ComfyUI-GGUF and ComfyUI_pytorch360convert node packs and ComfyUI ≥ v0.44.0.

Benchmark (HDR before the sun step, mean of 10): si-RMSE 0.129 / 0.250 / 0.484 (diffuse / matte / mirror),
colour error 2.64°, brightest light 38.6° from the real one; the original pipeline scored 0.146 / 0.280 / 0.513,
2.94°, 44.9°. Tested and rejected: a generic pole prompt (invents objects on the ground), stronger pole repair,
lanczos or blurred LuxDiT reference (slightly / clearly worse). The sun gate never adds a fake sun but finds
only 1 of 3 real suns, and is slightly negative on average; bypass nodes 50/51 to skip it.

## Settings that matter

- `env_resolution`: 128x256 is what the model was trained on. 256x512 also gives a clean panorama
  (same overall brightness, sun in the same place, but a softer/less peaked sun). 512x1024 was tested
  and produces blocky noise, so it is not offered.
- Real photos: `lora_scale` 0.8. Synthetic renders / object scans: 0.
- Steps: 50 for images, 40 for video. Guidance 2.5.
- Video: `luxdit_video`, frame count must be 4k+1 (trained on 7/17/25-frame clips; extra frames are dropped).
- Upstream pins `diffusers==0.32.0`, which cannot import under transformers 5.x (current ComfyUI
  images). This wrapper requires `diffusers>=0.42.1`, verified to run the upstream code.
- `.exr` is written with the `OpenEXR` package: OpenCV 5 pip wheels are built without EXR support.
- Upstream only documents Ampere-class GPUs (A100) on Linux. The bf16 transformer alone is ~11 GB.

## License

LuxDiT code and weights are under the **NVIDIA OneWay Noncommercial License** (research/evaluation use
only), and the CogVideoX base weights carry their own license. This wrapper downloads both from their
original sources at install/run time; using them is subject to those terms.
