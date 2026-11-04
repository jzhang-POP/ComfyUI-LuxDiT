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


def _dirs(h, w):
    """Unit direction per equirect pixel centre. u=0.5 is straight ahead (azimuth 0), v=0.5 the horizon."""
    v = (torch.arange(h, dtype=torch.float64) + 0.5) / h
    u = (torch.arange(w, dtype=torch.float64) + 0.5) / w
    el = (0.5 - v) * np.pi
    az = (u - 0.5) * 2 * np.pi
    el, az = torch.meshgrid(el, az, indexing="ij")
    return torch.stack([torch.cos(el) * torch.sin(az), torch.sin(el), torch.cos(el) * torch.cos(az)], -1)


def _solid_angle(h, w):
    v = (torch.arange(h, dtype=torch.float64) + 0.5) / h
    return (torch.sin(v * np.pi) * (2 * np.pi / w) * (np.pi / h)).view(h, 1).expand(h, w)


def _az_el(d):
    return float(np.degrees(np.arctan2(d[0], d[2]))), float(np.degrees(np.arcsin(np.clip(d[1], -1, 1))))


class LuxDiTSunTransfer:
    """Give a high-res HDR panorama LuxDiT's sun.

    LuxDiT estimates the sun's energy and colour from the photo, but only at
    128x256 / 256x512. Generated or reconstructed high-res HDRs often lack a real
    sun core. This node measures the sun lobe in LuxDiT's map (energy above the
    surrounding sky), optionally matches the high-res map's ambient level to
    LuxDiT's sky, and deposits a sun disc with that energy and colour into the
    high-res map, either where the high-res panorama shows its sun glow or in
    LuxDiT's sun direction.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hdr": ("IMAGE", {"tooltip": "High-res linear HDR panorama."}),
                "luxdit_hdr": ("IMAGE", {"tooltip": "LuxDiT HDR Merge output (rotate_to_camera on)."}),
                "placement": (["panorama glow", "luxdit direction"], {"default": "panorama glow",
                              "tooltip": "panorama glow: put the sun where the high-res image shows it, so lighting matches the visible background."}),
                "sun_radius_deg": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 5.0, "step": 0.05,
                                   "tooltip": "Angular radius of the sun disc. The real sun is ~0.27 deg; larger = softer shadows."}),
                "lobe_threshold": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.9, "step": 0.01,
                                   "tooltip": "LuxDiT pixels above this fraction of its peak (within 20 deg) count as the sun lobe."}),
                "match_ambient": ("BOOLEAN", {"default": True,
                                  "tooltip": "Scale the high-res map so its sky/ambient brightness matches LuxDiT's (sun excluded)."}),
                "sun_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("hdr", "report")
    FUNCTION = "run"
    CATEGORY = "LuxDiT/HDR tools"

    def run(self, hdr, luxdit_hdr, placement, sun_radius_deg, lobe_threshold, match_ambient, sun_strength):
        ref = luxdit_hdr[0, ..., :3].double().cpu()
        h, w = ref.shape[:2]
        lum = ref @ LUMA.double()
        dirs = _dirs(h, w)
        dom = _solid_angle(h, w)

        # --- sun lobe in LuxDiT ---
        peak_idx = int(lum.argmax())
        peak_dir = dirs.view(-1, 3)[peak_idx]
        cosang = (dirs @ peak_dir).clamp(-1, 1)
        ang = torch.rad2deg(torch.arccos(cosang))
        ring = (ang > 20) & (ang < 30)
        base = ref[ring].median(dim=0).values  # sky level around the sun, per channel
        lobe = (ang < 20) & (lum > lobe_threshold * lum.max())
        excess = (ref - base).clamp(min=0) * lobe[..., None]
        energy = (excess * dom[..., None]).sum((0, 1))  # per-channel, radiance * sr
        wsum = (excess @ LUMA.double() * dom)
        sun_dir_lux = (dirs * wsum[..., None]).sum((0, 1))
        sun_dir_lux = sun_dir_lux / sun_dir_lux.norm()

        # Ambient (sun removed) mean luminance of LuxDiT.
        ref_ambient = ref.clone()
        ref_ambient[lobe] = base
        mean_ref = float(((ref_ambient @ LUMA.double()) * dom).sum() / dom.sum())

        out, reports = [], []
        for i in range(hdr.shape[0]):
            x = hdr[i, ..., :3].double().cpu()
            H, W = x.shape[:2]
            # ambient match, measured at LuxDiT's resolution
            x_lo = _resize_wrap(x.permute(2, 0, 1)[None].float(), (h, w), "area")[0].permute(1, 2, 0).double()
            mean_x = float(((x_lo @ LUMA.double()) * dom).sum() / dom.sum())
            scale = mean_ref / max(mean_x, 1e-9) if match_ambient else 1.0
            x = x * scale

            # where to put the sun
            if placement == "panorama glow":
                blur = F.avg_pool2d(x_lo.permute(2, 0, 1)[None].float(), 5, stride=1, padding=2)[0]
                bl = (blur.permute(1, 2, 0).double() @ LUMA.double())
                bl[: int(h * 0.02)] = 0  # ignore the pole rows, often clipped sky
                sun_dir = dirs.view(-1, 3)[int(bl.argmax())]
            else:
                sun_dir = sun_dir_lux

            # deposit a disc with LuxDiT's sun energy
            D = _dirs(H, W)
            dOm = _solid_angle(H, W)
            a = torch.rad2deg(torch.arccos((D @ sun_dir).clamp(-1, 1)))
            px_deg = 360.0 / W
            wdisc = ((sun_radius_deg + 0.5 * px_deg - a) / px_deg).clamp(0, 1)  # anti-aliased edge
            disc_sr = float((wdisc * dOm).sum())
            radiance = energy * sun_strength / max(disc_sr, 1e-12)
            x = x + wdisc[..., None] * radiance

            az_l, el_l = _az_el(sun_dir_lux.numpy())
            az_p, el_p = _az_el(sun_dir.numpy())
            reports.append(
                f"LuxDiT sun: energy (lum) {float(energy @ LUMA.double()):.4f} sr-weighted, colour rgb "
                f"{(energy / energy.max()).numpy().round(3).tolist()}, dir az {az_l:.1f} el {el_l:.1f}\n"
                f"placed at az {az_p:.1f} el {el_p:.1f} ({placement}), radius {sun_radius_deg} deg, "
                f"disc radiance {float(radiance @ LUMA.double()):.1f}\n"
                f"ambient: LuxDiT {mean_ref:.4f} vs input {mean_x:.4f} -> scale x{scale:.3f}"
            )
            out.append(x.float())
        report = "\n".join(reports)
        print("[LuxDiT sun] " + report.replace("\n", " | "))
        return (torch.stack(out), report)


NODE_CLASS_MAPPINGS = {
    "LuxDiTHDRDetailUpscale": LuxDiTHDRDetailUpscale,
    "LuxDiTCalibrateHDR": LuxDiTCalibrateHDR,
    "LuxDiTSunTransfer": LuxDiTSunTransfer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LuxDiTHDRDetailUpscale": "LuxDiT HDR Detail Upscale (gain map)",
    "LuxDiTCalibrateHDR": "LuxDiT Calibrate HDR",
    "LuxDiTSunTransfer": "LuxDiT Sun Transfer",
}
