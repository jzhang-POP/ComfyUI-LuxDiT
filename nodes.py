import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np
import torch
from PIL import Image

import comfy.model_management as mm
import comfy.utils
import folder_paths

from . import luxdit_bridge

LUXDIT_DIR = os.path.join(folder_paths.models_dir, "luxdit")
BASE_DIR = os.path.join(LUXDIT_DIR, "CogVideoX-5b-I2V")
HF_LUXDIT = "nvidia/LuxDiT"
HF_BASE = "zai-org/CogVideoX-5b-I2V"  # formerly THUDM/CogVideoX-5b-I2V

# Mirrors configs/luxdit_base.yaml in the upstream repo.
TRANSFORMER_KWARGS = dict(
    additional_input_channels=None,
    additional_patch_embed_channels=48,
    additional_output_channels=32,
    use_learned_positional_embeddings=False,
    copy_weights=True,
    copy_temb_layer=True,
)
COND_LABELS = {"rgb": "vae", "env_nrm": "vae"}  # order matters: rgb, then env_nrm
ADDITIONAL_COND_LABELS = ["env_ldr", "env_log", "env_nrm"]
TARGET_LABELS = ["rgb"]
ADDITIONAL_TARGET_LABELS = ["env_ldr", "env_log"]
ENV_RESOLUTIONS = {
    "128x256 (native)": (128, 256),
    "256x512 (experimental)": (256, 512),
    "512x1024 (experimental)": (512, 1024),
}


def _download(repo_id, local_dir, patterns):
    from huggingface_hub import snapshot_download

    print(f"[LuxDiT] downloading {repo_id} {patterns} -> {local_dir}")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, allow_patterns=patterns)


def ensure_luxdit_weights(subdir):
    path = os.path.join(LUXDIT_DIR, subdir)
    if not os.path.isfile(os.path.join(path, "diffusion_pytorch_model.safetensors")):
        _download(HF_LUXDIT, LUXDIT_DIR, [f"{subdir}/*", "LICENSE.md"])
    return path


def ensure_base_weights():
    if not (
        os.path.isfile(os.path.join(BASE_DIR, "vae", "diffusion_pytorch_model.safetensors"))
        and os.path.isfile(os.path.join(BASE_DIR, "scheduler", "scheduler_config.json"))
    ):
        _download(HF_BASE, BASE_DIR, ["vae/*", "scheduler/*", "LICENSE"])
    return BASE_DIR


def _to_comfy(frames):
    """list of PIL images -> IMAGE tensor [F, H, W, 3] float32 in 0..1."""
    arr = np.stack([np.asarray(f.convert("RGB")) for f in frames]).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _tonemap(hdr, exposure):
    x = np.clip(hdr * (2.0 ** exposure), 0.0, None)
    x = x / (1.0 + x)
    return np.clip(x ** (1.0 / 2.2), 0.0, 1.0)


def write_exr(path, rgb):
    """Write an HxWx3 float32 linear RGB array as .exr.

    Uses the OpenEXR package; OpenCV 5 pip wheels ship without EXR support.
    """
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    try:
        import OpenEXR
    except ImportError:
        OpenEXR = None
    if OpenEXR is not None:
        header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
        with OpenEXR.File(header, {"RGB": rgb}) as f:
            f.write(path)
        return
    import cv2

    if not cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"[LuxDiT] could not write {path}: pip install OpenEXR")


def _seam_blending(img, blend_width):
    """Verbatim port of hdr_merger.seam_blending from the upstream repo."""
    height, width = img.shape[:2]
    result = img.copy()
    left_weights = np.linspace(1, 0.5, blend_width)
    right_weights = 1 - left_weights
    left_region = img[:, -blend_width:].copy()
    right_region = img[:, :blend_width].copy()
    for i in range(blend_width):
        result[:, width - blend_width + i] = (
            left_region[:, i] * left_weights[i] + right_region[:, blend_width - 1 - i] * right_weights[i]
        )
        result[:, blend_width - 1 - i] = (
            right_region[:, blend_width - 1 - i] * left_weights[i] + left_region[:, i] * right_weights[i]
        )
    return result


class LuxDiTModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["luxdit_image", "luxdit_video"], {"default": "luxdit_image",
                          "tooltip": "luxdit_image for a single image, luxdit_video for a frame sequence."}),
                "precision": (["bf16", "fp16"], {"default": "bf16"}),
                "load_lora": ("BOOLEAN", {"default": True,
                              "tooltip": "Real-scene LoRA. Its strength is set by lora_scale on the sampler (0 = off)."}),
                "offload": (["none", "model_cpu_offload"], {"default": "none",
                            "tooltip": "model_cpu_offload lowers VRAM use at the cost of speed."}),
            }
        }

    RETURN_TYPES = ("LUXDIT_PIPE",)
    RETURN_NAMES = ("luxdit_pipe",)
    FUNCTION = "load"
    CATEGORY = "LuxDiT"

    def load(self, model, precision, load_lora, offload):
        m = luxdit_bridge.load()
        from diffusers.schedulers import CogVideoXDPMScheduler

        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        device = mm.get_torch_device()
        model_dir = ensure_luxdit_weights(model)
        base_dir = ensure_base_weights()

        vae = m["vae"].AutoencoderKLCogVideoX.from_pretrained(base_dir, subfolder="vae", torch_dtype=dtype)
        transformer = m["transformer"].CustomCogVideoXTransformer3DModel.from_pretrained(
            model_dir, torch_dtype=dtype, **TRANSFORMER_KWARGS
        )
        scheduler = CogVideoXDPMScheduler.from_pretrained(base_dir, subfolder="scheduler")
        pipe = m["pipeline"].RGBXEnvCogVideoXPipeline(
            vae=vae, tokenizer=None, text_encoder=None, transformer=transformer, scheduler=scheduler
        )

        if load_lora:
            lora_dir = os.path.join(model_dir, "lora")
            pipe.load_lora_weights(lora_dir, weight_name="model.safetensors", adapter_name="env-lora")

        if offload == "model_cpu_offload":
            pipe.enable_model_cpu_offload(device=device)
        else:
            pipe.to(device)

        return ({"pipe": pipe, "model": model, "device": device},)


class LuxDiTSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "luxdit_pipe": ("LUXDIT_PIPE",),
                "images": ("IMAGE", {"tooltip": "One image, or a batch of video frames (1, 5, 9, ... 25 frames)."}),
                "resolution": (["480x720", "512x512"], {"default": "480x720",
                               "tooltip": "Input is resized and center-cropped to this. 512x512 suits object scans."}),
                "env_resolution": (list(ENV_RESOLUTIONS), {"default": "128x256 (native)",
                                   "tooltip": "Output panorama size. The model was trained at 128x256; larger sizes are experimental."}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200,
                          "tooltip": "Upstream uses 50 for images, 40 for video."}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "lora_scale": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                               "tooltip": "0.8 for real photos, 0 for synthetic renders / object scans."}),
                "seed": ("INT", {"default": 33, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "max_frames": ("INT", {"default": 25, "min": 1, "max": 49,
                               "tooltip": "Video model was trained on 7/17/25-frame clips."}),
            },
            "optional": {
                "camera_elevation": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0,
                                     "tooltip": "Camera pitch in degrees, if known."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("env_ldr", "env_log")
    FUNCTION = "sample"
    CATEGORY = "LuxDiT"

    def sample(self, luxdit_pipe, images, resolution, env_resolution, steps, guidance_scale, lora_scale, seed,
               max_frames, camera_elevation=0.0):
        render = luxdit_bridge.load()["render"]
        pipe = luxdit_pipe["pipe"]
        device = luxdit_pipe["device"]
        height, width = map(int, resolution.split("x"))

        frames = images[..., :3].detach().cpu().float().numpy()
        n = min(frames.shape[0], max_frames)
        if n > 1 and (n - 1) % 4 != 0:
            n = ((n - 1) // 4) * 4 + 1
            print(f"[LuxDiT] using the first {n} frames (frame count must be 4k+1)")
        frames = frames[:n]
        if n > 1 and luxdit_pipe["model"] != "luxdit_video":
            print("[LuxDiT] warning: multiple frames given to luxdit_image; luxdit_video is meant for sequences")

        rgb = np.stack([render.resize_crop(f, (height, width)) for f in frames], axis=0)

        env_nrm_raw = render.envmap_vec(ENV_RESOLUTIONS[env_resolution])
        if camera_elevation != 0:
            elevation = camera_elevation / 180 * np.pi
            c2w = torch.linalg.inv(render.get_cam_matrix(np.pi / 2, elevation))
            env_nrm = (env_nrm_raw.reshape(-1, 3) @ c2w[:3, :3]).reshape_as(env_nrm_raw)
        else:
            env_nrm = env_nrm_raw
        env_nrm = env_nrm * 0.5 + 0.5
        env_nrm = env_nrm[None, :].repeat(n, 1, 1, 1).numpy()

        example = {"rgb": rgb, "env_nrm": env_nrm}
        _, cond_images = pipe.example2input(example, ADDITIONAL_TARGET_LABELS, dict(COND_LABELS))

        mm.unload_all_models()
        mm.soft_empty_cache()

        pbar = comfy.utils.ProgressBar(steps)

        def on_step(_pipe, i, t, kwargs):
            mm.throw_exception_if_processing_interrupted()
            pbar.update(1)
            return {}

        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            pred = pipe(
                prompt=None,
                cond_images=cond_images,
                cond_mapping=dict(COND_LABELS),
                guidance_scale=guidance_scale,
                use_dynamic_cfg=False,
                num_inference_steps=steps,
                generator=generator,
                height=height,
                width=width,
                num_frames=n,
                additional_cond_labels=ADDITIONAL_COND_LABELS,
                attention_kwargs={"scale": lora_scale},
                denoise_type="additional_hidden_states",
                target_labels=TARGET_LABELS,
                additional_target_labels=ADDITIONAL_TARGET_LABELS,
                num_latents=len(ADDITIONAL_TARGET_LABELS),
                separate_timesteps=True,
                additional_rope_time_only=False,
                callback_on_step_end=on_step,
            ).frames

        env_ldr = _to_comfy(pred[0][0])
        env_log = _to_comfy(pred[1][0])
        mm.soft_empty_cache()
        return (env_ldr, env_log)


_HDR_MLP = None


def _hdr_model(device):
    global _HDR_MLP
    if _HDR_MLP is None:
        hdr = luxdit_bridge.load()["hdr"]
        path = ensure_luxdit_weights("hdr_merge_mlp")
        _HDR_MLP = hdr.HDR_MLP.from_pretrained(path).eval()
    return _HDR_MLP.to(device)


class LuxDiTHDRMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "env_ldr": ("IMAGE",),
                "env_log": ("IMAGE",),
                "rotate_to_camera": ("BOOLEAN", {"default": True,
                                     "tooltip": "Roll the map so its center faces the camera direction (upstream default)."}),
                "seam_blending": ("BOOLEAN", {"default": True, "tooltip": "Only applies when rotate_to_camera is on."}),
                "flip_horizontal": ("BOOLEAN", {"default": True, "tooltip": "Upstream default."}),
                "preview_exposure": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.25,
                                     "tooltip": "Only affects the tonemapped preview output."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("hdr", "preview")
    FUNCTION = "merge"
    CATEGORY = "LuxDiT"

    def merge(self, env_ldr, env_log, rotate_to_camera, seam_blending, flip_horizontal, preview_exposure):
        device = mm.get_torch_device()
        model = _hdr_model(device)
        hdr_out, previews = [], []
        with torch.no_grad():
            for ldr, log in zip(env_ldr, env_log):
                x_ldr = (ldr[..., :3].float() * 2.0 - 1.0).to(device)
                x_log = (log[..., :3].float() * 2.0 - 1.0).to(device)
                hdr = model(x_ldr, x_log).float().cpu().numpy()
                if rotate_to_camera:
                    width = hdr.shape[1]
                    if seam_blending:
                        hdr = _seam_blending(hdr, blend_width=int(width * 0.02))
                    hdr = np.roll(hdr, width // 2, axis=1)
                if flip_horizontal:
                    hdr = np.flip(hdr, axis=1)
                hdr = np.ascontiguousarray(hdr, dtype=np.float32)
                hdr_out.append(hdr)
                previews.append(_tonemap(hdr, preview_exposure).astype(np.float32))
        return (torch.from_numpy(np.stack(hdr_out)), torch.from_numpy(np.stack(previews)))


class LuxDiTSaveEXR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hdr": ("IMAGE", {"tooltip": "Linear HDR from LuxDiT HDR Merge (values above 1 are kept)."}),
                "filename_prefix": ("STRING", {"default": "luxdit/envmap"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "LuxDiT"

    def save(self, hdr, filename_prefix):
        h, w = hdr.shape[1], hdr.shape[2]
        out_dir = folder_paths.get_output_directory()
        full_dir, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir, w, h)
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)

        ui_images = []
        for img in hdr:
            rgb = np.ascontiguousarray(img[..., :3].float().cpu().numpy(), dtype=np.float32)
            name = f"{filename}_{counter:05}_"
            exr_path = os.path.join(full_dir, name + ".exr")
            write_exr(exr_path, rgb)
            print(f"[LuxDiT] saved {exr_path}")

            preview_name = f"luxdit_preview_{filename}_{counter:05}_.png"
            Image.fromarray((_tonemap(rgb, 0.0) * 255).round().astype(np.uint8)).save(
                os.path.join(temp_dir, preview_name)
            )
            ui_images.append({"filename": preview_name, "subfolder": "", "type": "temp"})
            counter += 1

        return {"ui": {"images": ui_images}}


NODE_CLASS_MAPPINGS = {
    "LuxDiTModelLoader": LuxDiTModelLoader,
    "LuxDiTSampler": LuxDiTSampler,
    "LuxDiTHDRMerge": LuxDiTHDRMerge,
    "LuxDiTSaveEXR": LuxDiTSaveEXR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LuxDiTModelLoader": "LuxDiT Model Loader",
    "LuxDiTSampler": "LuxDiT Sampler",
    "LuxDiTHDRMerge": "LuxDiT HDR Merge",
    "LuxDiTSaveEXR": "LuxDiT Save EXR",
}
