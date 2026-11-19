"""LuxDiT Photo Paste: put the original input photo back into the generated panorama.

The 360 generator redraws the photo's region (softer detail, small drift; it keeps the photo centred but picks its
own zoom and tilt). The photo and the panorama are views from the same point, so one homography maps the photo
onto a pinhole view of the panorama's front. We find it with SIFT + RANSAC (as ComfyUI-PixelDriftFix does for
same-framing edits), then sample the photo for every panorama pixel in its footprint. The generator also moves
parts of the scene by a few pixels, which one homography cannot follow, so around the border: optical flow moves the
generated pixels onto the photo's features, the cut is placed where the two agree, and fine detail fades in over a
wider band than colour (the photo is sharper than the generated surroundings). Works on the sRGB LDR panorama (before the HDR merge, so the merge carries the photo's detail).
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


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _erode(m, r):
    r = int(round(r))
    if r <= 0:
        return m.copy()
    return cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))


def _signed_dist(m):
    """Signed distance (pixels) to the edge of a 0/1 mask: positive inside."""
    m = m.astype(np.uint8)
    return (cv2.distanceTransform(m, cv2.DIST_L2, 5) - cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)).astype(np.float32)


def _border_flow(wp, view, foot, size):
    """Displacement F (view pixels) such that view(p + F(p)) ~ wp(p): where the generator drew the photo's content.
    Measured inside the footprint, smoothed, carried a little way outside it and faded to zero."""
    fl = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(_gray8(wp), _gray8(view), None)
    lim = 0.04 * size
    mag = np.linalg.norm(fl, axis=2, keepdims=True)
    fl = fl * np.minimum(1.0, lim / np.maximum(mag, 1e-6))
    valid = _erode(foot, 3).astype(np.float32)
    def fill(sig):
        num = cv2.GaussianBlur(fl * valid[..., None], (0, 0), sig)
        den = cv2.GaussianBlur(valid, (0, 0), sig)
        return num / np.maximum(den, 1e-6)[..., None], den
    F, den = fill(0.02 * size)
    Fw, _ = fill(0.08 * size)  # carries the drift further out than the fine fill reaches
    k = np.clip(den / 0.3, 0, 1)[..., None]
    F = k * F + (1 - k) * Fw
    outside = cv2.distanceTransform((1 - foot).astype(np.uint8), cv2.DIST_L2, 5)
    return (F * _smoothstep(1 - outside / (0.25 * size))[..., None]).astype(np.float32)


def _seam(gen, photo, allow, band, prefer=0.03, step=3):
    """Where to cut between the generated view and the warped photo, inside `allow` (the photo's usable area).
    Walks inward from the edge of `allow` along each of its boundary points; at each point picks the depth
    (0..band px) where the two images agree, preferring the outermost one (so the photo keeps as much area as it
    can), with a smooth cut along the boundary (dynamic programming around the loop).
    Returns the photo side as a 0/1 mask."""
    cs, _ = cv2.findContours(allow.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return allow
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    c = c[::step]
    n = len(c)
    # inward normals from the smoothed contour tangent
    sm = lambda a: cv2.GaussianBlur(np.concatenate([a[-8:], a, a[:8]])[:, None], (1, 0), 3)[8:-8, 0]
    cx, cy = sm(c[:, 0]), sm(c[:, 1])
    tx, ty = np.roll(cx, -1) - np.roll(cx, 1), np.roll(cy, -1) - np.roll(cy, 1)
    nx, ny = -ty, tx
    nn = np.maximum(np.hypot(nx, ny), 1e-6)
    nx, ny = nx / nn, ny / nn
    m = allow.astype(np.float32)
    test = cv2.remap(m, (cx + 4 * nx)[:, None], (cy + 4 * ny)[:, None], cv2.INTER_LINEAR)[:, 0]
    if test.mean() < 0.5:  # normals point outward: flip
        nx, ny = -nx, -ny
    diff = cv2.GaussianBlur(np.abs(gen - photo).sum(2), (0, 0), 1.5)
    d = np.arange(int(band) + 1, dtype=np.float32)
    X = (cx[:, None] + nx[:, None] * d[None]).astype(np.float32)
    Y = (cy[:, None] + ny[:, None] * d[None]).astype(np.float32)
    cost = cv2.remap(diff, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) + prefer * d[None] / max(band, 1)
    # DP around the loop (two laps; the second lap's path is used), depth may change by <=1 px per station
    D = len(d)
    lap = np.concatenate([cost, cost])
    acc = lap[0].copy()
    back = np.zeros(lap.shape, np.int16)
    idx = np.arange(D)
    for t in range(1, len(lap)):
        prev = np.stack([np.roll(acc, 1), acc, np.roll(acc, -1)])
        prev[0, 0] = np.inf
        prev[2, -1] = np.inf
        k = prev.argmin(0)
        back[t] = idx + k - 1
        acc = prev[k, idx] + lap[t]
    path = np.empty(len(lap), np.int64)
    path[-1] = int(acc.argmin())
    for t in range(len(lap) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    depth = path[n:].astype(np.float32)
    depth = cv2.GaussianBlur(np.concatenate([depth[-8:], depth, depth[:8]])[:, None], (1, 0), 2)[8:-8, 0]
    # every pixel of `allow`: its distance to the edge vs the cut depth at its nearest boundary station
    st = np.ones(allow.shape, np.uint8)
    xi = np.clip(np.round(c[:, 0]).astype(int), 0, allow.shape[1] - 1)
    yi = np.clip(np.round(c[:, 1]).astype(int), 0, allow.shape[0] - 1)
    st[yi, xi] = 0
    _, lab = cv2.distanceTransformWithLabels(st, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(lab.max() + 1, np.float32)
    lut[lab[yi, xi]] = depth
    inside = cv2.distanceTransform(allow.astype(np.uint8), cv2.DIST_L2, 5)
    return (allow.astype(bool) & (inside >= lut[lab])).astype(np.uint8)


def _view_to_pano(u, v, H, W):
    """View-plane pixel coords -> equirect (col, row)."""
    f = _view_rays()
    x, y = (u + 0.5 - VIEW_W / 2) / f, -(v + 0.5 - VIEW_H / 2) / f
    lon = np.degrees(np.arctan2(x, 1.0))
    lat = np.degrees(np.arctan2(y, np.sqrt(x * x + 1)))
    return (lon / 360 + 0.5) * W - 0.5, (90 - lat) / 180 * H - 0.5


def paste_photo(pano, photo, feather=0.03, min_inliers=30, align=True, seam=True, detail_fade=0.10,
                return_parts=False):
    """pano: sRGB equirect (H, W, 3) in [0, 1]; photo: sRGB (h, w, 3). Returns (pano with the photo pasted,
    colour blend mask (H, W) in [0, 1], report[, parts]). Widths are fractions of the footprint's size
    (square root of its area in the front view).
      feather: width of the colour/brightness blend across the cut.
      align: optical-flow nudge of the generated pixels around the photo onto the photo's features.
      seam: put the cut where the photo and the panorama agree (graph cut) instead of along the photo's frame.
      detail_fade: width over which fine detail goes from the panorama's to the photo's, inside the cut."""
    pano = pano[..., :3].astype(np.float32)
    photo = photo[..., :3].astype(np.float32)
    H, W = pano.shape[:2]
    view = _front_view(pano)
    Hm, why = _homography(photo, view, min_inliers)
    if Hm is None:
        out = (pano, np.zeros((H, W), np.float32), f"photo paste skipped ({why})")
        return out + ({},) if return_parts else out
    h, w = photo.shape[:2]
    # ---- view-plane fields (front view resolution)
    foot = (cv2.warpPerspective(np.ones((h, w), np.float32), Hm, (VIEW_W, VIEW_H)) > 0.999).astype(np.uint8)
    size = float(np.sqrt(foot.sum()))
    wp = cv2.warpPerspective(photo, Hm, (VIEW_W, VIEW_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    flow = _border_flow(wp, view, foot, size) if align else np.zeros((VIEW_H, VIEW_W, 2), np.float32)
    gu, gv = np.meshgrid(np.arange(VIEW_W, dtype=np.float32), np.arange(VIEW_H, dtype=np.float32))
    view_al = cv2.remap(view, gu + flow[..., 0], gv + flow[..., 1], cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE) if align else view
    wl = max(feather, 1e-3) * size
    allow = _erode(foot, wl / 2 + 2)
    side = _seam(view_al, wp, allow, 0.15 * size) if seam else allow
    sd = _signed_dist(side)
    m_lo = _smoothstep((sd + wl / 2) / wl) * foot
    m_hi = _smoothstep(sd / (detail_fade * size)) if detail_fade > 0 else m_lo
    m_hi = np.minimum(m_hi, m_lo)
    # ---- panorama pixels of the front hemisphere -> view plane -> photo pixels
    f = _view_rays()
    r0, r1, c0, c1 = 0, H, W // 4, 3 * W // 4  # |azimuth| < 90 deg
    el = np.radians(90 - (np.arange(r0, r1) + 0.5) / H * 180)[:, None]
    az = np.radians(((np.arange(c0, c1) + 0.5) / W - 0.5) * 360)[None, :]
    x, y, z = np.cos(el) * np.sin(az), np.sin(el) * np.ones_like(az), np.cos(el) * np.cos(az)
    z = np.maximum(z, 1e-6)
    u = (f * x / z + VIEW_W / 2 - 0.5).astype(np.float32)
    v = (VIEW_H / 2 - f * y / z - 0.5).astype(np.float32)
    samp = lambda a: cv2.remap(a, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    lo, hi = samp(m_lo), samp(m_hi)
    Hi = np.linalg.inv(Hm)
    den = Hi[2, 0] * u + Hi[2, 1] * v + Hi[2, 2]
    px = (Hi[0, 0] * u + Hi[0, 1] * v + Hi[0, 2]) / den
    py = (Hi[1, 0] * u + Hi[1, 1] * v + Hi[1, 2]) / den
    lo[den <= 0] = 0
    hi[den <= 0] = 0
    sub = pano[r0:r1, c0:c1]
    gen = sub
    if align:  # generated pixels around the photo, moved onto the photo's features
        fx, fy = samp(flow[..., 0]), samp(flow[..., 1])
        moved = (np.abs(fx) + np.abs(fy)) > 1e-3
        if moved.any():
            cc, rr = _view_to_pano(u + fx, v + fy, H, W)
            g2 = cv2.remap(pano, cc.astype(np.float32), rr.astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_WRAP)
            gen = np.where(moved[..., None], g2, sub)
    sample = cv2.remap(photo, px.astype(np.float32), py.astype(np.float32), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    sig = 2.0 * W / 4096  # detail = what a ~2 px (at 4K) blur removes
    lg, lp = cv2.GaussianBlur(gen, (0, 0), sig), cv2.GaussianBlur(sample, (0, 0), sig)
    blend = lg + lo[..., None] * (lp - lg) + (gen - lg) + hi[..., None] * ((sample - lp) - (gen - lg))
    out = pano.copy()
    out[r0:r1, c0:c1] = np.where(lo[..., None] > 0, blend, gen)
    mask = np.zeros((H, W), np.float32)
    mask[r0:r1, c0:c1] = lo
    rep = f"photo pasted ({why}, {100 * float((mask > 0.5).mean()):.1f}% of the panorama)"
    if not return_parts:
        return out, mask, rep
    dmask = np.zeros((H, W), np.float32)
    dmask[r0:r1, c0:c1] = hi
    return out, mask, rep, {"mask_low": mask, "mask_detail": dmask, "flow_view": flow, "seam_view": side}


class LuxDiTPhotoPaste:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "panorama": ("IMAGE", {"tooltip": "Generated sRGB panorama (equirect), e.g. the 4K LDR fed to the merge."}),
            "photo": ("IMAGE", {"tooltip": "The original input photo."}),
            "feather": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 0.3, "step": 0.005,
                        "tooltip": "Width of the colour blend across the cut, as a fraction of the photo's size in "
                                   "the panorama."}),
            "min_inliers": ("INT", {"default": 30, "min": 8, "max": 500,
                            "tooltip": "Matched points needed to trust the alignment; below it the panorama is "
                                       "returned unchanged."}),
        }, "optional": {
            "align": ("BOOLEAN", {"default": True,
                      "tooltip": "Nudge the generated pixels around the photo onto the photo's features (optical "
                                 "flow), so objects crossing the border line up."}),
            "seam": ("BOOLEAN", {"default": True,
                     "tooltip": "Cut where the photo and the panorama agree instead of along the photo's frame."}),
            "detail_fade": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 0.4, "step": 0.01,
                            "tooltip": "Width over which fine detail goes from the panorama's to the photo's, so the "
                                       "sharp photo does not sit in the softer panorama as a hard rectangle."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("panorama", "mask", "report")
    FUNCTION = "run"
    CATEGORY = "LuxDiT"

    def run(self, panorama, photo, feather=0.03, min_inliers=30, align=True, seam=True, detail_fade=0.10):
        outs, masks, reps = [], [], []
        for b in range(panorama.shape[0]):
            ph = photo[min(b, photo.shape[0] - 1)].float().cpu().numpy()
            o, m, r = paste_photo(panorama[b].float().cpu().numpy(), ph, feather, min_inliers, align, seam,
                                  detail_fade)
            outs.append(torch.from_numpy(o))
            masks.append(torch.from_numpy(m))
            reps.append(r)
            print(f"[LuxDiT] {r}")
        return torch.stack(outs), torch.stack(masks), "\n".join(reps)


NODE_CLASS_MAPPINGS = {"LuxDiTPhotoPaste": LuxDiTPhotoPaste}
NODE_DISPLAY_NAME_MAPPINGS = {"LuxDiTPhotoPaste": "LuxDiT Photo Paste"}
