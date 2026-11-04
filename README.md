# ComfyUI-LuxDiT

ComfyUI nodes for [NVIDIA LuxDiT](https://github.com/nv-tlabs/LuxDiT) (lighting estimation with a video
diffusion transformer): image or video frames in, HDR environment map (`.exr`) out.

## Nodes

| Node | What it does |
| --- | --- |
| **LuxDiT Model Loader** | Loads `luxdit_image` or `luxdit_video` (+ optional real-scene LoRA) and the CogVideoX-5b-I2V VAE/scheduler. Downloads weights on first use. |
| **LuxDiT Sampler** | Takes an `IMAGE` (one image, or a batch of video frames) and returns the two 256x512 panoramas the model predicts: tonemapped LDR and log-intensity. |
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

## Settings that matter

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
