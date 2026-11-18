"""LuxDiT Photo Paste: put the original input photo back into the generated panorama.

The 360 generator redraws the photo's region (softer detail, small drift; it keeps the photo centred but picks its
own zoom and tilt). The photo and the panorama are views from the same point, so one homography maps the photo
onto a pinhole view of the panorama's front. We find it with SIFT + RANSAC (as ComfyUI-PixelDriftFix does for
same-framing edits), then sample the photo for every panorama pixel in its footprint and blend it in with a
feathered mask. Works on the sRGB LDR panorama (before the HDR merge, so the merge carries the photo's detail).
"""
import cv2
import numpy as np
import torch

VIEW_W, VIEW_H, VIEW_VFOV = 1440, 960, 90.0
MAX_PHOTO_SIDE = 1280


def _view_rays(w=VIEW_W, h=VIEW_H, vfov=VIEW_VFOV):
    """Focal length (pixels) of the front view."""
    return (h / 2) / np.tan(np.radians(vfov) / 2)


def _front_view(pano, w=VIEW_W, h=VIEW_H, vfov=VIEW_VFOV):
    """Pinhole view of the panorama centre (azimuth 0, elevation 0); same convention as the bench renderer."""
    H, W = pano.shape[:2]
    f = _view_rays(w, h, vfov)
    j, i = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x, y = (j + 0.5 - w / 2) / f, -(i + 0.5 - h / 2) / f
    n = np.sqrt(x * x + y * y + 1)
    lon, lat = np.degrees(np.arctan2(x / n, 1 / n)), np.degrees(np.arcsin(y / n))
    col = (lon / 360 + 0.5) * W - 0.5
    row = (90 - lat) / 180 * H - 0.5
    return cv2.remap(pano.astype(np.float32), col.astype(np.float32), row.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _gray8(x):
    return cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def _homography(photo, view, min_inliers):
    """photo -> view homography (pixel coords), or (None, reason)."""
    s = min(1.0, MAX_PHOTO_SIDE / max(photo.shape[:2]))
    ph = cv2.resize(photo, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else photo
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(_gray8(ph), None)
    k2, d2 = sift.detectAndCompute(_gray8(view), None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None, "too few features"
    good = [m for m, n in cv2.BFMatcher().knnMatch(d1, d2, k=2) if m.distance < 0.8 * n.distance]
    if len(good) < 8:
        return None, f"{len(good)} matches"
    src = np.float32([k1[m.queryIdx].pt for m in good]) / s
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    ni = int(inl.sum()) if inl is not None else 0
    if Hm is None or ni < min_inliers:
        return None, f"{ni} inliers < {min_inliers}"
    h, w = photo.shape[:2]
    corners = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]])[None], Hm)[0]
    area = cv2.contourArea(corners)
    if not cv2.isContourConvex(corners) or area < 0.02 * VIEW_W * VIEW_H or area > 0.98 * VIEW_W * VIEW_H:
        return None, "implausible footprint"
    return Hm, f"{ni} inliers"


def paste_photo(pano, photo, feather=0.06, min_inliers=30):
    """pano: sRGB equirect (H, W, 3) in [0, 1]; photo: sRGB (h, w, 3). Returns (pano with the photo pasted,
    blend mask (H, W) in [0, 1], report). feather: blend ramp width as a fraction of the photo's shorter side."""
    pano = pano[..., :3].astype(np.float32)
    photo = photo[..., :3].astype(np.float32)
    H, W = pano.shape[:2]
    Hm, why = _homography(photo, _front_view(pano), min_inliers)
    if Hm is None:
        return pano, np.zeros((H, W), np.float32), f"photo paste skipped ({why})"
    # panorama pixels of the front hemisphere -> view plane -> photo pixels
    f = _view_rays()
    r0, r1, c0, c1 = 0, H, W // 4, 3 * W // 4  # |azimuth| < 90 deg
    el = np.radians(90 - (np.arange(r0, r1) + 0.5) / H * 180)[:, None]
    az = np.radians(((np.arange(c0, c1) + 0.5) / W - 0.5) * 360)[None, :]
    x, y, z = np.cos(el) * np.sin(az), np.sin(el) * np.ones_like(az), np.cos(el) * np.cos(az)
    z = np.maximum(z, 1e-6)
    u = f * x / z + VIEW_W / 2 - 0.5
    v = VIEW_H / 2 - f * y / z - 0.5
    Hi = np.linalg.inv(Hm)
    den = Hi[2, 0] * u + Hi[2, 1] * v + Hi[2, 2]
    px = (Hi[0, 0] * u + Hi[0, 1] * v + Hi[0, 2]) / den
    py = (Hi[1, 0] * u + Hi[1, 1] * v + Hi[1, 2]) / den
    h, w = photo.shape[:2]
    edge = np.minimum.reduce([px, w - px, py, h - py]) / (max(feather, 1e-6) * min(h, w))
    m = np.clip(edge, 0, 1)
    m = m * m * (3 - 2 * m)  # smoothstep
    m[den <= 0] = 0
    sample = cv2.remap(photo, px.astype(np.float32), py.astype(np.float32), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    out = pano.copy()
    mask = np.zeros((H, W), np.float32)
    mask[r0:r1, c0:c1] = m
    sub = out[r0:r1, c0:c1]
    out[r0:r1, c0:c1] = np.where(m[..., None] > 0, sub + m[..., None] * (sample - sub), sub)
    return out, mask, f"photo pasted ({why}, {100 * float((mask > 0.5).mean()):.1f}% of the panorama)"


class LuxDiTPhotoPaste:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "panorama": ("IMAGE", {"tooltip": "Generated sRGB panorama (equirect), e.g. the 4K LDR fed to the merge."}),
            "photo": ("IMAGE", {"tooltip": "The original input photo."}),
            "feather": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.3, "step": 0.01,
                        "tooltip": "Width of the blend at the photo's border, as a fraction of its shorter side."}),
            "min_inliers": ("INT", {"default": 30, "min": 8, "max": 500,
                            "tooltip": "Matched points needed to trust the alignment; below it the panorama is "
                                       "returned unchanged."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("panorama", "mask", "report")
    FUNCTION = "run"
    CATEGORY = "LuxDiT"

    def run(self, panorama, photo, feather=0.06, min_inliers=30):
        outs, masks, reps = [], [], []
        for b in range(panorama.shape[0]):
            ph = photo[min(b, photo.shape[0] - 1)].float().cpu().numpy()
            o, m, r = paste_photo(panorama[b].float().cpu().numpy(), ph, feather, min_inliers)
            outs.append(torch.from_numpy(o))
            masks.append(torch.from_numpy(m))
            reps.append(r)
            print(f"[LuxDiT] {r}")
        return torch.stack(outs), torch.stack(masks), "\n".join(reps)


NODE_CLASS_MAPPINGS = {"LuxDiTPhotoPaste": LuxDiTPhotoPaste}
NODE_DISPLAY_NAME_MAPPINGS = {"LuxDiTPhotoPaste": "LuxDiT Photo Paste"}
