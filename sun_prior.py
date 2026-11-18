"""LuxDiT Sun (prior) — drop-in replacement for LuxDiT Sun (gated) in the photo-to-HDRI workflow.

Decides whether the panorama shows a sun with a small classifier on the LDR panorama's sky
(trained on 527 Poly Haven HDRIs outside the LuxDiT test list), places it at the panorama's
brightest glow, removes whatever partial sun DiffHDR made there, and adds a sun disc carrying
`target_share` of all the light (default 0.55; real visible suns: median 0.70), coloured by elevation.
Defaults = configuration B, confirmed on fresh random draws (docs/results/2026-11-18-sun-v4.md).

Self-contained (numpy + OpenCV); copy into ComfyUI-LuxDiT and register in __init__.py.
Equirect convention as hdr_tools: u=0.5 is azimuth 0, v=0.5 the horizon.
"""
import cv2
import numpy as np
import torch

LUMA = np.array([0.2126, 0.7152, 0.0722])
CORE_DEG = 5.0

# Logistic regression on LDR sky features (bench/lab/sun_classifier.json, LDR simulated at median 0.06).
CLF = {
    "features": ["sky_lum", "sky_lum_std", "sky_sat", "sky_blue", "sky_clip", "n_blobs", "blob_r", "blob_el",
                 "blob_compact"],
    "mu": [0.456932, 0.172432, 0.283459, -0.000157, 0.06229, 2.469203, 2.004156, 11.967686, 0.430772],
    "sd": [0.152664, 0.08009, 0.15054, 0.002448, 0.121103, 2.126431, 1.725877, 48.311299, 0.262378],
    "coef": [-1.679199, 1.026831, 0.802821, -0.060776, -1.500576, -1.066176, 0.085186, 0.902421, -0.253089],
    "intercept": -1.328776,
}
# v3: logistic regression on the same sky features + LuxDiT's own sun estimate, trained on 240 non-test Poly Haven
# HDRIs run through the panorama stage (bench/lab/sun_classifier_v3.json). Blind AUC 0.71 vs 0.66 for CLF.
CLF_V3 = {"features": ["sky_lum", "sky_lum_std", "sky_sat", "sky_blue", "sky_clip", "n_blobs", "blob_r", "blob_el", "blob_compact", "L_share", "L_ratio", "L_el"],
          "logs": ["n_blobs", "L_ratio"],
          "mu": [0.503121322, 0.171153819, 0.280913178, 0.000528491, 0.064977064, 2.968238582, 2.20246905, 16.582035518, 0.423194767, 0.07334292, 5.818285077, 24.588867188],
          "sd": [0.17112586, 0.057020099, 0.160444988, 0.003338789, 0.125066508, 1.950089467, 1.647592464, 40.701883502, 0.203107936, 0.110480274, 1.651904541, 20.793458013],
          "coef": [-0.0584029216723945, 0.2923401551054543, 0.18851432637468668, -0.4960382676494713, -0.4697882066867487, -0.6795513347882751, -0.2923081224874163, 0.23873957497666345, 0.3026808760782104, 0.4646787878900412, 0.05265135711773312, 0.33132640079375375],
          "intercept": -0.35925120899346646, "threshold": 0.5}
# Median colour of real suns by elevation (deg), from the same 527 HDRIs.
SUN_RGB_EL = ((5.0, (1.307, 0.961, 0.447)), (17.0, (1.127, 0.993, 0.689)), (40.0, (1.05, 0.993, 0.901)))


# ---------------------------------------------------------------- geometry
def _el_az(h, w):
    return (0.5 - (np.arange(h) + 0.5) / h) * np.pi, ((np.arange(w) + 0.5) / w - 0.5) * 2 * np.pi


def _dirs(h, w):
    el, az = np.meshgrid(*_el_az(h, w), indexing="ij")
    return np.stack([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)], -1)


def _solid_angle(h, w):
    v = (np.arange(h) + 0.5) / h
    return np.repeat((np.sin(v * np.pi) * (2 * np.pi / w) * (np.pi / h))[:, None], w, axis=1)


def _az_el(d):
    return float(np.degrees(np.arctan2(d[0], d[2]))), float(np.degrees(np.arcsin(np.clip(d[1], -1, 1))))


def _angles(h, w, d):
    return np.degrees(np.arccos(np.clip(_dirs(h, w) @ d, -1, 1)))


def _luma(x):
    return x[..., :3] @ LUMA


# ---------------------------------------------------------------- user coordinates
# Panorama coordinates u, v in [0, 1] (left->right, top->bottom of the equirect image). Azimuth 0 is the
# centre of the panorama = where the photo was placed; +90 is to the right; elevation +90 is straight up.
def uv_to_dir(u, v):
    az, el = (u - 0.5) * 2 * np.pi, (0.5 - v) * np.pi
    return np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])


def dir_to_uv(d):
    az, el = _az_el(d)
    return float(az / 360.0 + 0.5), float(0.5 - el / 180.0)


def angles_to_dir(azimuth_deg, elevation_deg):
    return uv_to_dir(azimuth_deg / 360.0 + 0.5, 0.5 - elevation_deg / 180.0)


def parse_points(text):
    """'[[u, v], ...]' (what the click widget writes) or 'u,v'. Values are panorama fractions in [0, 1]."""
    import json
    text = (text or "").strip()
    if not text:
        return []
    try:
        val = json.loads(text)
    except ValueError:
        try:
            val = [float(x) for x in text.replace(";", ",").split(",")]
        except ValueError:
            raise ValueError(f"cannot read points {text[:60]!r}; expected [[u, v], ...]") from None
    if not isinstance(val, list):
        raise ValueError(f"cannot read points {text[:60]!r}; expected [[u, v], ...]")
    if val and isinstance(val[0], (int, float)):
        val = [val[i:i + 2] for i in range(0, len(val), 2)]
    pts = []
    for p in val:
        if not (isinstance(p, (list, tuple)) and len(p) == 2):
            raise ValueError(f"point {p!r} is not a [u, v] pair")
        u, v = float(p[0]), float(p[1])
        if not (0 <= u <= 1 and 0 <= v <= 1):
            raise ValueError(f"point {p!r} is outside the panorama (u, v must be in [0, 1])")
        pts.append((u, v))
    return pts


def sun_from_shadow(points, elevation_deg=None):
    """Sun direction from clicks on the panorama, assuming flat ground below the camera.

    2 clicks: base of an upright object, then the tip of its shadow -> azimuth (elevation from `elevation_deg`).
    3 clicks: base, top of the object, tip of its shadow -> azimuth and elevation.
    Returns (direction, elevation_came_from_clicks)."""
    if len(points) not in (2, 3):
        raise ValueError("shadow needs 2 clicks (object base, shadow tip) or 3 (base, top, shadow tip)")
    base, tip = uv_to_dir(*points[0]), uv_to_dir(*points[-1])
    for name, d in (("object base", base), ("shadow tip", tip)):
        if d[1] > -0.01:
            raise ValueError(f"the {name} must be on the ground (below the horizon)")
    B, T = base / -base[1], tip / -tip[1]          # camera 1 unit above the ground plane y = -1
    shadow = np.array([T[0] - B[0], T[2] - B[2]])
    length = float(np.hypot(*shadow))
    if length < 1e-6:
        raise ValueError("object base and shadow tip are the same point")
    away = -shadow / length                          # the sun is opposite the shadow
    if len(points) == 3:
        t = uv_to_dir(*points[1])
        txz = np.array([t[0], t[2]])
        k = float(txz @ np.array([B[0], B[2]]) / max(txz @ txz, 1e-12))
        height = k * t[1] + 1.0
        if height <= 0:
            raise ValueError("the object top must be above its base")
        el, from_clicks = float(np.degrees(np.arctan2(height, length))), True
    else:
        if elevation_deg is None:
            raise ValueError("2 shadow clicks give the azimuth only; set the sun elevation")
        el, from_clicks = float(elevation_deg), False
    c = np.cos(np.radians(el))
    return np.array([c * away[0], np.sin(np.radians(el)), c * away[1]]), from_clicks


def kelvin_rgb(kelvin):
    """Linear sRGB colour of a black body at `kelvin`, normalised to luminance 1
    (CIE 1931 observer, multi-lobe fit of Wyman, Sloan & Shirley 2013)."""
    kelvin = max(float(kelvin), 1000.0)  # below ~1000 K a black body is invisible deep red: not a sun colour
    lam = np.arange(380.0, 781.0, 5.0)

    def g(mu, s1, s2):
        return np.exp(-0.5 * ((lam - mu) / np.where(lam < mu, s1, s2)) ** 2)

    xb = 1.056 * g(599.8, 37.9, 31.0) + 0.362 * g(442.0, 16.0, 26.7) - 0.065 * g(501.1, 20.4, 26.2)
    yb = 0.821 * g(568.8, 46.9, 40.5) + 0.286 * g(530.9, 16.3, 31.1)
    zb = 1.217 * g(437.0, 11.8, 36.0) + 0.681 * g(459.0, 26.0, 13.8)
    lm = lam * 1e-9
    planck = lm ** -5 / np.expm1(1.4388e-2 / (lm * float(kelvin)))
    xyz = np.array([(planck * xb).sum(), (planck * yb).sum(), (planck * zb).sum()])
    rgb = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]) @ xyz
    rgb = np.maximum(rgb, 1e-4 * rgb.max())
    return rgb / float(rgb @ LUMA)


# ---------------------------------------------------------------- sun detector
def _blobs(ldr, clip=0.95, max_radius_deg=6.0, min_el_deg=-2.0, min_compact=0.3):
    h, w = ldr.shape[:2]
    n, lab = cv2.connectedComponents((ldr[..., :3].min(-1) >= clip).astype(np.uint8), connectivity=8)
    el_r, az_r = _el_az(h, w)
    dom = _solid_angle(h, w)[:, 0]
    out = []
    flat = lab.ravel()
    order = np.argsort(flat, kind="stable")             # group pixels by label once (no per-label full scan)
    bounds = np.searchsorted(flat[order], np.arange(n + 1))
    for k in range(1, n):
        idx = order[bounds[k]:bounds[k + 1]]
        rr, cc = idx // w, idx % w
        area = float(dom[rr].sum())
        r_eq = float(np.degrees(np.sqrt(area / np.pi)))
        if r_eq > max_radius_deg:
            continue
        el, az = el_r[rr], az_r[cc]
        dv = np.stack([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)], -1)
        c = (dv * dom[rr][:, None]).sum(0)
        c /= np.linalg.norm(c)
        r_max = float(np.degrees(np.arccos(np.clip(dv @ c, -1, 1))).max()) + 0.5 * 360.0 / w
        compact = (r_eq / max(r_max, 1e-6)) ** 2
        cel = _az_el(c)[1]
        if cel < min_el_deg or compact < min_compact:
            continue
        out.append({"el": cel, "dir": c, "radius_deg": r_eq, "area_sr": area, "compact": compact})
    return sorted(out, key=lambda b: -b["area_sr"])


def sky_features(ldr):
    """ldr: sRGB [0,1] equirect, any size (evaluated at 512x1024)."""
    if ldr.shape[:2] != (512, 1024):
        ldr = cv2.resize(ldr.astype(np.float32), (1024, 512), interpolation=cv2.INTER_AREA)
    el = _el_az(*ldr.shape[:2])[0]
    sky = ldr[el > np.radians(10)]
    lum = sky @ LUMA
    mx, mn = sky.max(-1), sky.min(-1)
    blobs = _blobs(ldr)
    b = blobs[0] if blobs else None
    return {"sky_lum": float(lum.mean()), "sky_lum_std": float(lum.std()),
            "sky_sat": float(np.median((mx - mn) / np.maximum(mx, 1e-6))),
            "sky_blue": float(np.median(sky[:, 2] - sky[:, 0])), "sky_clip": float((mn >= 0.95).mean()),
            "n_blobs": len(blobs), "blob_r": b["radius_deg"] if b else 0.0, "blob_el": b["el"] if b else -99.0,
            "blob_compact": b["compact"] if b else 0.0}


def sun_probability(ldr, clf=CLF):
    f = sky_features(ldr)
    x = np.array([f[k] for k in clf["features"]], float)
    x[clf["features"].index("n_blobs")] = np.log1p(x[clf["features"].index("n_blobs")])
    z = (x - np.array(clf["mu"])) / np.array(clf["sd"])
    return float(1 / (1 + np.exp(-(z @ np.array(clf["coef"]) + clf["intercept"])))), f


def luxdit_sun_features(lux, core_deg=CORE_DEG, ring_deg=10.0):
    """LuxDiT's sun as the lab measures it: brightest 3x3-blurred direction at 256x512 (top 2% rows ignored),
    its ratio to the median, its elevation, and the share of all light within core_deg above the ring sky."""
    lo = _lo(lux)
    lum = _luma(lo)
    bl = cv2.blur(lum, (3, 3), borderType=cv2.BORDER_REFLECT)
    bl[: int(lo.shape[0] * 0.02)] = 0
    r, c = np.unravel_index(int(np.argmax(bl)), bl.shape)
    d = _dirs(*lo.shape[:2])[r, c]
    a = _angles(*lo.shape[:2], d)
    ring = (a > core_deg) & (a < ring_deg)
    base = np.median(lo[ring], axis=0) if ring.any() else np.zeros(3)
    dom = _solid_angle(*lo.shape[:2])
    ex = np.clip(lo - base, 0, None) * (a <= core_deg)[..., None]
    share = float((ex * dom[..., None]).sum((0, 1)) @ LUMA) / max(float((lum * dom).sum()), 1e-12)
    return {"L_ratio": float(bl[r, c] / max(np.median(lum), 1e-12)), "L_share": share, "L_el": _az_el(d)[1]}


def sun_probability_v3(ldr, lux, clf=CLF_V3):
    f = {**sky_features(ldr), **luxdit_sun_features(lux)}
    x = np.array([f[k] for k in clf["features"]], float)
    for k in clf["logs"]:
        x[clf["features"].index(k)] = np.log1p(x[clf["features"].index(k)])
    z = (x - np.array(clf["mu"])) / np.array(clf["sd"])
    return float(1 / (1 + np.exp(-(z @ np.array(clf["coef"]) + clf["intercept"])))), f


# ---------------------------------------------------------------- sun layer
def _lo(m, h=256, w=512):
    return m[..., :3].astype(np.float64) if m.shape[:2] == (h, w) else cv2.resize(
        m[..., :3].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA).astype(np.float64)


def brightest_dir(m, pole_frac=0.02):
    """Direction of the brightest glow, or None when nothing stands out above the median (flat map)."""
    lo = _lo(m)
    lum = _luma(lo)
    bl = cv2.blur(lum, (3, 3), borderType=cv2.BORDER_REFLECT)
    bl[: int(lo.shape[0] * pole_frac)] = 0
    r, c = np.unravel_index(int(np.argmax(bl)), bl.shape)
    if not bl[r, c] > np.median(lum) * 1.001:
        return None
    return _dirs(*lo.shape[:2])[r, c]


def _finite(a):
    """Replace NaN / inf (e.g. fp16 overflow upstream) so one bad pixel cannot move the sun or poison the map."""
    a = np.asarray(a, np.float32)
    bad = ~np.isfinite(a)
    if not bad.any():
        return a, 0
    good = a[~bad]
    return np.where(bad, np.where(np.isposinf(a), good.max() if good.size else 0.0, 0.0), a).astype(np.float32), int(bad.sum())


def _np(t):
    return t.float().cpu().numpy()


def bright_candidates(m, k=10, min_el=-1.0, pole_frac=0.02, nms_deg=5.0):
    lo = _lo(m)
    bl = cv2.blur(_luma(lo), (3, 3), borderType=cv2.BORDER_REFLECT)
    bl[: int(lo.shape[0] * pole_frac)] = 0
    bl[np.degrees(_el_az(*lo.shape[:2])[0]) < min_el] = 0
    peaks = (bl == cv2.dilate(bl, np.ones((9, 9), np.uint8))) & (bl > 0)
    rr, cc = np.nonzero(peaks)
    D = _dirs(*lo.shape[:2])
    kept = []
    for i in np.argsort(-bl[rr, cc]):
        d = D[rr[i], cc[i]]
        if all(np.degrees(np.arccos(np.clip(d @ x["dir"], -1, 1))) >= nms_deg for x in kept):
            kept.append({"dir": d, "value": float(bl[rr[i], cc[i]])})
            if len(kept) == k:
                break
    return kept


def glow_candidates(m, k=3, sep_deg=20.0):
    """Up to k glow directions at least sep_deg apart, brightest first (shown on the preview as click targets)."""
    kept = []
    for c in bright_candidates(m, k=12):
        if all(np.degrees(np.arccos(np.clip(c["dir"] @ d, -1, 1))) >= sep_deg for d in kept):
            kept.append(c["dir"])
            if len(kept) == k:
                break
    return kept


def glow_distinctness(m, sep_deg=20.0):
    """Brightest glow / best other glow at least sep_deg away. Near 1 = no distinct sun glow."""
    c = bright_candidates(m)
    if not c:
        return 0.0
    others = [x["value"] for x in c[1:] if np.degrees(np.arccos(np.clip(x["dir"] @ c[0]["dir"], -1, 1))) >= sep_deg]
    return float(c[0]["value"] / others[0]) if others else float("inf")


def sun_color(el_deg):
    els = [e for e, _ in SUN_RGB_EL]
    cols = np.array([c for _, c in SUN_RGB_EL])
    return np.array([np.interp(el_deg, els, cols[:, k]) for k in range(3)])


def set_sun(m, d, target, color, radius_deg=0.5, return_parts=False):
    """Within CORE_DEG of d, dim only what is brighter than the surrounding sky (the generated sun and its glow;
    darker detail such as trees stays), then add a disc with `target` of all light.
    return_parts: also return the sun-free sky and the sun's energy (luminance integrated over its disc)."""
    h, w = m.shape[:2]
    a = _angles(h, w, d)
    ring = (a > CORE_DEG) & (a < CORE_DEG * 1.5)
    out = m[..., :3].astype(np.float64).copy()
    lum = out @ LUMA
    cap = float(np.percentile(lum[ring], 90.0)) if ring.any() else float(np.median(lum))
    core = a <= CORE_DEG
    out[core] *= np.minimum(1.0, cap / np.maximum(lum[core], 1e-12))[:, None]
    dom = _solid_angle(h, w)
    rest = float((_luma(out) * dom).sum())
    e_sun = target / (1 - target) * rest
    px = 360.0 / w
    wd = np.clip((radius_deg + 0.5 * px - a) / px, 0, 1)
    if wd.sum() == 0:
        wd = (a == a.min()).astype(float)
    c = np.asarray(color, float)
    sky = out.astype(np.float32) if return_parts else None
    out += wd[..., None] * (c / float(c @ LUMA) * e_sun / float((wd * dom).sum()))
    if return_parts:
        return out.astype(np.float32), sky, float(e_sun)
    return out.astype(np.float32)


def snap_to_glow(m, d, snap_deg):
    """Brightest (3x3-blurred) point of m within snap_deg of d; d itself when nothing is there."""
    if snap_deg <= 0:
        return d
    lo = _lo(m)
    bl = cv2.blur(_luma(lo), (3, 3), borderType=cv2.BORDER_REFLECT)
    near = _angles(*lo.shape[:2], d) <= snap_deg
    if not near.any():
        return d
    bl = np.where(near, bl, -np.inf)
    r, c = np.unravel_index(int(np.argmax(bl)), bl.shape)
    return _dirs(*lo.shape[:2])[r, c]


# ---------------------------------------------------------------- LuxDiT low-frequency blend
def _sh_basis(h, w):
    d = _dirs(h, w)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    return np.stack([0.282095 * np.ones_like(x), 0.488603 * y, 0.488603 * z, 0.488603 * x,
                     1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z,
                     0.546274 * (x * x - y * y)], -1)


def _sh_project(m):
    h, w = m.shape[:2]
    return np.einsum("hwk,hwc->kc", _sh_basis(h, w) * _solid_angle(h, w)[..., None], m[..., :3].astype(np.float64))


def _remove_sun(m, d, radius_deg):
    h, w = m.shape[:2]
    a = _angles(h, w, d)
    out = m[..., :3].copy()
    ring = (a > radius_deg) & (a < radius_deg * 1.5)
    out[a <= radius_deg] = np.median(m[ring][:, :3] if ring.any() else m[..., :3].reshape(-1, 3), axis=0)
    return out


def _peak_ratio(m):
    lo = _lo(m)
    lum = _luma(lo)
    bl = cv2.blur(lum, (3, 3), borderType=cv2.BORDER_REFLECT)
    bl[: int(lo.shape[0] * 0.02)] = 0
    return float(bl.max() / max(np.median(lum), 1e-12))


def _wrap_blur(x, frac=1 / 24.0):
    """Gaussian blur of an equirect map, wrapping in azimuth (sigma = frac of the width)."""
    h, w = x.shape[:2]
    s = max(1.0, w * frac)
    p = min(w, int(3 * s) + 1)
    xp = np.concatenate([x[:, -p:], x, x[:, :p]], 1).astype(np.float32)
    return cv2.GaussianBlur(xp, (0, 0), s, borderType=cv2.BORDER_REFLECT)[:, p:p + w].astype(np.float64)


def _sh_reconstruct(c, h, w):
    return np.einsum("hwk,kc->hwc", _sh_basis(h, w), c)


def blend_toward_luxdit(m, lux, alpha, lo=0.7, hi=4.0, iters=3):
    """Move m's order-2 (diffuse-scale) lighting `alpha` of the way toward LuxDiT's, suns excluded.
    The change is a smooth per-channel gain on m, clamped to [lo, hi]: dark detail keeps its contrast and
    nothing is dimmed below `lo` (an added correction used to floor dark ground into a brown smudge).
    The gain is solved on a 512x1024 area-downsample and applied to the full map in float32, 256 rows at a time."""
    if alpha == 0:
        return m[..., :3].astype(np.float32)
    ms = m[..., :3].astype(np.float32)
    if ms.shape[0] > 512:
        ms = cv2.resize(ms, (1024, 512), interpolation=cv2.INTER_AREA)
    dm, dl = brightest_dir(ms), brightest_dir(lux)
    m_amb = _remove_sun(ms, dm, 5.0) if dm is not None and _peak_ratio(ms) >= 300 else ms
    l_amb = _remove_sun(lux, dl, 15.0) if dl is not None and _peak_ratio(lux) >= 30 else lux
    amb = m_amb[..., :3].astype(np.float64)
    cm, cr = _sh_project(amb), _sh_project(l_amb)
    cr = cr * (cm[0] / np.maximum(cr[0], 1e-12))
    want = cm + alpha * (cr - cm)
    sh, sw = amb.shape[:2]
    base = np.maximum(_wrap_blur(amb), 1e-6 * max(float(amb.mean()), 1e-12))
    g, cur = np.ones_like(base), cm
    for _ in range(iters):
        g = np.clip(g + _sh_reconstruct(want - cur, sh, sw) / base, lo, hi)
        g = np.clip(_wrap_blur(g), lo, hi)
        cur = _sh_project(amb * g)
    h, w = m.shape[:2]
    g = g.astype(np.float32)
    if (sh, sw) != (h, w):
        g = np.clip(cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR), lo, hi)
    out = np.empty((h, w, 3), np.float32)
    for r0 in range(0, h, 256):
        rows = slice(r0, min(h, r0 + 256))
        out[rows] = m[rows, :, :3].astype(np.float32) * g[rows]
    return out


# ---------------------------------------------------------------- preview
def preview_image(m, sun_dir=None, auto_dir=None, points=(), shadow=False, size=(512, 1024), candidates=()):
    """Tone-mapped panorama (sRGB, [0,1]) with the sun, the automatic pick and the user's clicks drawn on it.
    It covers the whole panorama, so a click at fraction (x, y) of the image is the panorama point (u, v)."""
    h, w = size
    lo = cv2.resize(m[..., :3].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    x = lo * (0.18 / max(float(np.median(_luma(lo))), 1e-9))
    x = x / (1.0 + x)
    img = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)
    img = np.ascontiguousarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    cv2.line(img, (0, h // 2), (w - 1, h // 2), (90, 90, 90), 1)
    for az in (-180, -90, 0, 90):
        xx = int((az / 360 + 0.5) * w) % w
        cv2.line(img, (xx, h // 2 - 6), (xx, h // 2 + 6), (160, 160, 160), 1)
        cv2.putText(img, f"{az:+d}" if az else "0 photo", (xx + 3, h // 2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 200, 200), 1, cv2.LINE_AA)
    px = lambda d: (int(dir_to_uv(d)[0] * w) % w, min(h - 1, int(dir_to_uv(d)[1] * h)))
    for k, cdir in enumerate(candidates):  # other glows the user may click (the node snaps to them)
        if sun_dir is not None and np.degrees(np.arccos(np.clip(cdir @ sun_dir, -1, 1))) < 3:
            continue
        cc = px(cdir)
        cv2.circle(img, cc, 8, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.circle(img, cc, 8, (235, 235, 235), 1, cv2.LINE_AA)
        for col, th in (((20, 20, 20), 3), ((235, 235, 235), 1)):
            cv2.putText(img, chr(ord("a") + k), (cc[0] + 10, cc[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, th,
                        cv2.LINE_AA)
    if auto_dir is not None:
        cv2.circle(img, px(auto_dir), 14, (170, 170, 170), 1, cv2.LINE_AA)
    pts = [(min(w - 1, int(u * w)), min(h - 1, int(v * h))) for u, v in points]
    if shadow and len(pts) >= 2:
        cv2.line(img, pts[0], pts[-1], (0, 220, 255), 1, cv2.LINE_AA)
        if len(pts) == 3:
            cv2.line(img, pts[0], pts[1], (0, 220, 255), 1, cv2.LINE_AA)
    for k, p in enumerate(pts):
        cv2.circle(img, p, 4, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.putText(img, str(k + 1), (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1,
                    cv2.LINE_AA)
    if sun_dir is not None:
        c = px(sun_dir)
        cv2.circle(img, c, 11, (255, 210, 0), 2, cv2.LINE_AA)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv2.line(img, (c[0] + 14 * dx, c[1] + 14 * dy), (c[0] + 22 * dx, c[1] + 22 * dy), (255, 210, 0), 2)
    return img.astype(np.float32) / 255.0


STRENGTH_PRESETS = {"custom (target_share)": None, "hazy (0.45)": 0.45, "typical (0.65)": 0.65, "clear (0.80)": 0.80}


def effective_share(target_share, preset="custom (target_share)", sun_ev=0.0):
    """Preset (or target_share), then sun_ev stops: +1 doubles the sun relative to the sky."""
    s = STRENGTH_PRESETS.get(preset) or target_share
    r = s / (1 - s) * 2.0 ** sun_ev
    return float(min(0.97, r / (1 + r)))


SUN_MODES = ["auto", "force sun", "no sun", "keep generated"]
DIRECTIONS = ["auto", "click", "shadow", "angles"]


class LuxDiTSunPrior:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "hdr": ("IMAGE", {"tooltip": "Linear HDR panorama (output of LuxDiT HDR Detail Upscale)."}),
            "ldr_high": ("IMAGE", {"tooltip": "The sRGB panorama fed to the merge (4K LDR)."}),
            "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                          "tooltip": "Built-in detector probability needed to add a sun in 'auto' mode when no photo "
                                     "check is connected (0 = always, 1 = never)."}),
            "target_share": ("FLOAT", {"default": 0.55, "min": 0.05, "max": 0.95, "step": 0.01,
                             "tooltip": "Sun strength, as the share of all light the sun carries. Real visible suns: "
                                        "median 0.70 (clear sky ~0.75-0.8, hazy ~0.4-0.5). 0.55 scored best when the "
                                        "sun is placed automatically; raise it when you set the direction yourself."}),
            "sun_radius_deg": ("FLOAT", {"default": 0.27, "min": 0.1, "max": 5.0, "step": 0.01,
                               "tooltip": "Angular RADIUS of the sun disc = shadow softness; brightness is unchanged. "
                                          "Default 0.27 = the real sun (diameter 0.53, the default 'Angle' / 'Sun Size' "
                                          "in Blender, Karma, Unreal): same lighting as 0.5 and a smaller benchmark "
                                          "mirror penalty. Raise it for softer shadows."}),
        }, "optional": {
            "luxdit_hdr": ("IMAGE", {"tooltip": "LuxDiT HDR Merge output (rotate_to_camera on). Enables the blend."}),
            "luxdit_blend": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                             "tooltip": "How far to move the low-frequency lighting toward LuxDiT's (0 = off)."}),
            "direction_fallback": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 3.0, "step": 0.05,
                                   "tooltip": "Auto direction: use LuxDiT's sun direction when the panorama's brightest "
                                              "glow is less than this many times brighter than the next one (0 = never)."}),
            "sun_mode": (SUN_MODES, {"default": "auto",
                         "tooltip": "auto: the detector decides (unless you set a direction). force sun / no sun: you "
                                    "decide. keep generated: "
                                    "leave whatever sun DiffHDR made (only the LuxDiT blend applies)."}),
            "direction": (DIRECTIONS, {"default": "auto",
                          "tooltip": "auto: panorama glow (LuxDiT if ambiguous). click: click the sun on the preview. "
                                     "shadow: click an object's base, its top, then the tip of its shadow (or base + "
                                     "tip and set the elevation). angles: type azimuth / elevation."}),
            "points": ("STRING", {"default": "", "multiline": False,
                       "tooltip": "Clicks on the preview as [[u, v], ...] (0-1 panorama fractions). Filled by clicking "
                                  "the preview; shift-click adds a point; you can also type them."}),
            "sun_azimuth": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                            "tooltip": "Degrees. 0 = straight ahead (centre of the photo), +90 = right, 180 = behind."}),
            "sun_elevation": ("FLOAT", {"default": 30.0, "min": -10.0, "max": 90.0, "step": 1.0,
                              "tooltip": "Degrees above the horizon (angles mode, and 2-click shadow mode)."}),
            "snap_deg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 15.0, "step": 0.5,
                         "tooltip": "Click mode: move the sun to the brightest point within this many degrees of the "
                                    "click (0 = exactly where clicked)."}),
            "strength_preset": (list(STRENGTH_PRESETS), {"default": "custom (target_share)",
                                "tooltip": "Sun strength presets (share of all light; real suns: 10-90% range 0.45-0.87). "
                                           "custom uses target_share."}),
            "sun_ev": ("FLOAT", {"default": 0.0, "min": -4.0, "max": 4.0, "step": 0.25,
                       "tooltip": "Stops on top of the share: +1 doubles the sun relative to the sky, -1 halves it."}),
            "p_sunny": ("FLOAT", {"forceInput": True, "tooltip": "From LuxDiT Photo Check: probability that the photo "
                                  "is lit by direct sunlight. When connected it replaces the sky detector."}),
            "p_indoor": ("FLOAT", {"forceInput": True, "tooltip": "From LuxDiT Photo Check: probability that the photo "
                                   "is indoors. Indoor scenes get no automatic sun and no LuxDiT blend."}),
            "detector": (["auto", "v3", "v1"], {"default": "auto",
                         "tooltip": "Built-in sun detector (used when no photo check is connected). v3 also reads "
                                    "LuxDiT's sun and needs luxdit_hdr; auto = v3 when it is connected, else v1."}),
            "luxdit_blend_outdoor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                     "tooltip": "LuxDiT blend used instead of luxdit_blend when the photo check says "
                                                "the scene is outdoors (it helps outdoors and hurts indoors/at night)."}),
            "photo_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                "tooltip": "p_sunny needed to add a sun in auto mode."}),
            "indoor_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                 "tooltip": "p_indoor above which the scene is treated as indoor."}),
            "sun_kelvin": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20000.0, "step": 100.0,  # 1..999 -> 1000
                           "tooltip": "Sun colour temperature. 0 = automatic from the sun's elevation "
                                      "(measured on real HDRIs: ~3300 K near the horizon, ~5500 K high)."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("hdr", "report", "preview", "sun_json", "sky_hdr")
    OUTPUT_TOOLTIPS = ("HDR with the sun.", "What the node decided and why.",
                       "Tone-mapped panorama with the sun and your clicks; click it to place the sun.",
                       "Sun as JSON (direction, share, irradiance, colour) to build a sun lamp in a DCC.",
                       "The same HDR without the sun disc, for use with a separate sun lamp.")
    FUNCTION = "run"
    CATEGORY = "LuxDiT/HDR tools"

    def run(self, hdr, ldr_high, threshold, target_share, sun_radius_deg, **kw):
        res = self.compute(hdr, ldr_high, threshold, target_share, sun_radius_deg, **kw)
        ui = _save_previews(res[2])
        return {"ui": {"sun_preview": ui}, "result": res} if ui else res  # shown by js/sun_picker.js

    def compute(self, hdr, ldr_high, threshold, target_share, sun_radius_deg, luxdit_hdr=None, luxdit_blend=0.25,
                direction_fallback=1.1, sun_mode="auto", direction="auto", points="", sun_azimuth=0.0,
                sun_elevation=30.0, snap_deg=3.0, sun_kelvin=0.0, strength_preset="custom (target_share)", sun_ev=0.0,
                p_sunny=None, p_indoor=None, photo_threshold=0.5, indoor_threshold=0.5, luxdit_blend_outdoor=0.5,
                detector="auto"):
        import json
        target_share = effective_share(target_share, strength_preset, sun_ev)
        out, skies, previews, infos, reports = [], [], [], [], []
        note0 = ""
        try:
            pts = parse_points(points)
        except ValueError as e:
            pts, note0 = [], f" | points ignored ({e})"
            direction = "auto" if direction in ("click", "shadow") else direction
        if direction in ("click", "shadow") and not pts:
            note0 += f" | no clicks yet, {direction} -> auto" if not note0 else ""
            direction = "auto"
        for i in range(hdr.shape[0]):
            m, nbad = _finite(_np(hdr[i, ..., :3]))
            note = note0 + (f" | {nbad} non-finite pixels replaced" if nbad else "")
            lux = None
            if luxdit_hdr is not None:
                lux = _finite(_np(luxdit_hdr[min(i, luxdit_hdr.shape[0] - 1), ..., :3]))[0]
            indoor = p_indoor is not None and p_indoor >= indoor_threshold
            alpha = luxdit_blend_outdoor if p_indoor is not None else luxdit_blend
            if lux is not None and alpha > 0 and not indoor:  # LuxDiT's lighting helps outdoors, hurts indoors
                m = blend_toward_luxdit(m, lux, alpha)
            ldr = _finite(_np(ldr_high[min(i, ldr_high.shape[0] - 1), ..., :3]))[0]
            use_v3 = lux is not None and detector in ("auto", "v3")
            p, _ = sun_probability_v3(ldr, lux) if use_v3 else sun_probability(ldr)
            det = "detector v3" if use_v3 else "detector v1"
            auto_d, auto_how = brightest_dir(m), "panorama glow"
            if lux is not None and direction_fallback > 0 and (auto_d is None or glow_distinctness(m) < direction_fallback):
                auto_d, auto_how = brightest_dir(lux), "LuxDiT direction"
            d, how = auto_d, auto_how
            if direction == "click":
                d, how = snap_to_glow(m, uv_to_dir(*pts[-1]), snap_deg), "your click" + (
                    f", snapped within {snap_deg:g} deg" if snap_deg > 0 else "")
            elif direction == "angles":
                d, how = angles_to_dir(sun_azimuth, sun_elevation), "your angles"
            elif direction == "shadow":
                try:
                    d, from_clicks = sun_from_shadow(pts, sun_elevation)
                    how = "shadow clicks" if from_clicks else "shadow azimuth + your elevation"
                except ValueError as e:
                    note += f" | shadow ignored ({e})"
                    direction = "auto"  # a failed construction must not force a sun
            if sun_mode == "keep generated":
                want, why = False, f"kept generated sun ({det} p={p:.2f})"
            elif sun_mode == "no sun":
                want, why = False, f"NO SUN (user) | {det} p={p:.2f}"
            elif sun_mode == "force sun":
                want, why = True, f"forced by user, {det} p={p:.2f}"
            elif direction != "auto":  # a direction given by the user means they want a sun there
                want, why = True, f"direction given by user, {det} p={p:.2f}"
            elif indoor:
                want, why = False, f"NO SUN | indoor scene (photo check p_indoor={p_indoor:.2f})"
            elif p_sunny is not None:
                want, why = p_sunny >= photo_threshold, f"photo check p_sunny={p_sunny:.2f}"
                if not want:
                    why = f"NO SUN | {why} < {photo_threshold}"
            else:
                want, why = p >= threshold, f"{det} p={p:.2f}"
            shadow_pts = direction == "shadow"
            if want and d is None:
                want, why = False, f"NO SUN | {why} but no glow to place it"
            if not want:
                out.append(torch.from_numpy(m))
                skies.append(torch.from_numpy(m))
                if sun_mode == "auto" and not why.startswith("NO SUN"):
                    why = f"NO SUN | {why} < {threshold}"
                reports.append(why + note)
                infos.append({"sun": False, "detector_p": round(p, 3)})
                previews.append(preview_image(m, None, auto_d, pts, shadow_pts, candidates=glow_candidates(m)))
                continue
            az, el = _az_el(d)
            color = kelvin_rgb(sun_kelvin) if sun_kelvin > 0 else sun_color(el)
            res, sky, e_sun = set_sun(m, d, target_share, color, sun_radius_deg, return_parts=True)
            out.append(torch.from_numpy(res))
            skies.append(torch.from_numpy(sky))
            col = np.asarray(color, float) / float(np.asarray(color, float) @ LUMA)
            infos.append({"sun": True, "azimuth_deg": round(az, 2), "elevation_deg": round(el, 2),
                          "direction_xyz": [round(float(v), 5) for v in d], "share": float(target_share),
                          "irradiance": e_sun, "rgb": [round(float(v), 4) for v in col],
                          "lamp_color": [round(float(v), 4) for v in col / col.max()],
                          "lamp_strength": e_sun * float(col.max()),
                          "angular_diameter_deg": 2 * float(sun_radius_deg), "source": how, "detector_p": round(p, 3),
                          "convention": "azimuth 0 = photo centre (panorama centre), +90 = right, 180 = behind; "
                                        "elevation above the horizon; direction_xyz = (right, up, forward); "
                                        "irradiance and lamp_strength in the HDR's own units (sun luminance "
                                        "integrated over its disc; lamp_strength x lamp_color = sun irradiance)"})
            reports.append(f"SUN {why} | az {az:.0f} el {el:.0f} ({how}) | share {target_share:.2f}" + note)
            previews.append(preview_image(res, d, auto_d if how != auto_how else None, pts, shadow_pts,
                                          candidates=glow_candidates(m)))
        report = "\n".join(reports)
        print("[LuxDiT sun prior] " + report.replace("\n", " || "))
        return (torch.stack(out), report, torch.from_numpy(np.stack(previews)), json.dumps(infos), torch.stack(skies))


def _save_previews(previews):
    """Write the previews to ComfyUI's temp folder so the node shows them (and the click widget can use them)."""
    try:
        import folder_paths
    except ImportError:  # outside ComfyUI (tests)
        return None
    import os
    import uuid
    out = []
    for p in previews.numpy():
        name = f"luxdit_sun_{uuid.uuid4().hex[:10]}.png"
        cv2.imwrite(os.path.join(folder_paths.get_temp_directory(), name),
                    cv2.cvtColor((np.clip(p, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        out.append({"filename": name, "subfolder": "", "type": "temp"})
    return out


class LuxDiTSaveText:
    """Save a STRING (e.g. sun_json) to the output folder as <prefix>_00001_.json, numbered like ComfyUI saves."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"forceInput": True}),
                             "filename_prefix": ("STRING", {"default": "photo2hdri_v4/sun"})}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "LuxDiT/HDR tools"

    def save(self, text, filename_prefix):
        import os
        import re
        import folder_paths
        root = folder_paths.get_output_directory()
        filename_prefix = filename_prefix.strip() or "sun"
        folder, base = os.path.split(os.path.join(root, os.path.normpath(filename_prefix)))
        base = base or "sun"
        if os.path.commonpath([os.path.realpath(folder), os.path.realpath(root)]) != os.path.realpath(root):
            raise ValueError("filename_prefix must stay inside the output folder")
        os.makedirs(folder, exist_ok=True)
        pat = re.compile(re.escape(base) + r"_(\d+)_\.json$")
        n = max([int(m.group(1)) for f in os.listdir(folder) if (m := pat.match(f))] + [0]) + 1
        name = f"{base}_{n:05d}_.json"
        with open(os.path.join(folder, name), "w") as f:
            f.write(text)
        return {"ui": {"text": [name]}}


NODE_CLASS_MAPPINGS = {"LuxDiTSunPrior": LuxDiTSunPrior, "LuxDiTSaveText": LuxDiTSaveText}
NODE_DISPLAY_NAME_MAPPINGS = {"LuxDiTSunPrior": "LuxDiT Sun (prior)", "LuxDiTSaveText": "LuxDiT Save Text (JSON)"}
