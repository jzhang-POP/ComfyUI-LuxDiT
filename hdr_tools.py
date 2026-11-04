"""HDR helper nodes: build a high-res HDR from a low-res HDR + high-res LDR, and
calibrate an HDR's overall exposure against LuxDiT's estimate."""

import numpy as np
import torch
import torch.nn.functional as F

LUMA = torch.tensor([0.2126, 0.7152, 0.0722])


def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _bchw(x):
    return x[..., :3].permute(0, 3, 1, 2).float()


def _bhwc(x):
    return x.permute(0, 2, 3, 1).contiguous()


def _resize_wrap(x, size, mode):
    """Resize an equirect BCHW tensor, padding horizontally so the 360° seam stays continuous."""
    h, w = x.shape[-2:]
    pad = max(2, w // 64)
    xp = torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)
    th, tw = size
    tpad = round(pad * tw / w)
    if mode == "area":
        y = F.interpolate(xp, size=(th, tw + 2 * tpad), mode="area")
    else:
        y = F.interpolate(xp, size=(th, tw + 2 * tpad), mode="bilinear", align_corners=False)
    return y[..., tpad:tpad + tw]


def _luma(x):
    return (x * LUMA.to(x.device).view(1, 3, 1, 1)).sum(1, keepdim=True)


class LuxDiTHDRDetailUpscale:
    """hdr_hi = linear(ldr_hi) * upsample(Y(hdr_lo) / Y(downsample(linear(ldr_hi))))

    The ratio (a smooth gain map) carries the dynamic range from the low-res HDR;
    the high-res LDR carries the detail. Where the high-res LDR is clipped, the
    upsampled HDR is used directly so the sun keeps its true intensity and colour.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hdr_low": ("IMAGE", {"tooltip": "Linear HDR panorama (e.g. 2048x1024 from DiffHDR)."}),
                "ldr_high": ("IMAGE", {"tooltip": "sRGB panorama of the same view at the target size (e.g. 4096x2048)."}),
                "clip_threshold": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01,
                                   "tooltip": "LDR values above this count as clipped and take the HDR's pixels."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("hdr_high",)
    FUNCTION = "run"
    CATEGORY = "LuxDiT/HDR tools"

    def run(self, hdr_low, ldr_high, clip_threshold):
        out = []
        for i in range(ldr_high.shape[0]):
            lo = _bchw(hdr_low[min(i, hdr_low.shape[0] - 1)][None])
            ldr = _bchw(ldr_high[i][None]).clamp(0, 1)
            lin = srgb_to_linear(ldr)
            H, W = lin.shape[-2:]
            h, w = lo.shape[-2:]

            lin_lo = _resize_wrap(lin, (h, w), "area")
            eps = 1e-4
            gain = (_luma(lo) + eps) / (_luma(lin_lo) + eps)
            gain_up = _resize_wrap(gain.log(), (H, W), "bilinear").exp()
            detail = lin * gain_up

            hdr_up = _resize_wrap(lo, (H, W), "bilinear")
            clipped = (ldr.max(dim=1, keepdim=True).values - clip_threshold) / max(1e-3, 1.0 - clip_threshold)
            mask = clipped.clamp(0, 1)
            mask = _resize_wrap(_resize_wrap(mask, (h, w), "area"), (H, W), "bilinear").clamp(0, 1)
            mask = torch.maximum(mask, clipped.clamp(0, 1))

            hdr = detail * (1 - mask) + hdr_up * mask
            out.append(_bhwc(hdr.clamp(min=0))[0])
        return (torch.stack(out),)


class LuxDiTCalibrateHDR:
    """Scale an HDR panorama so its average brightness matches LuxDiT's estimate,
    and report how the two compare (mean and sun peak)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hdr": ("IMAGE",),
                "luxdit_hdr": ("IMAGE", {"tooltip": "Output of LuxDiT HDR Merge (rotate_to_camera on)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("hdr", "report")
    FUNCTION = "run"
    CATEGORY = "LuxDiT/HDR tools"

    def run(self, hdr, luxdit_hdr):
        x = _bchw(hdr)
        ref = _bchw(luxdit_hdr[:1])
        h, w = ref.shape[-2:]
        x_lo = _resize_wrap(x, (h, w), "area")
        # Solid-angle weights for equirect rows.
        lat = torch.linspace(0.5, h - 0.5, h) / h * np.pi
        wts = torch.sin(lat).view(1, 1, h, 1)
        mean_x = float((_luma(x_lo) * wts).sum() / (wts.sum() * w))
        mean_ref = float((_luma(ref) * wts).sum() / (wts.sum() * w))
        scale = mean_ref / max(mean_x, 1e-8)
        report = (
            f"mean luminance: input {mean_x:.4f}, LuxDiT {mean_ref:.4f} -> scale x{scale:.3f}\n"
            f"peak luminance: input {float(_luma(x).max()):.2f} (x{scale:.3f} = {float(_luma(x).max()) * scale:.2f}), "
            f"LuxDiT {float(_luma(ref).max()):.2f}"
        )
        print("[LuxDiT calibrate] " + report.replace("\n", " | "))
        return (_bhwc(x * scale), report)


NODE_CLASS_MAPPINGS = {
    "LuxDiTHDRDetailUpscale": LuxDiTHDRDetailUpscale,
    "LuxDiTCalibrateHDR": LuxDiTCalibrateHDR,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LuxDiTHDRDetailUpscale": "LuxDiT HDR Detail Upscale (gain map)",
    "LuxDiTCalibrateHDR": "LuxDiT Calibrate HDR",
}
