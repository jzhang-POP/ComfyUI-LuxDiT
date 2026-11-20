# LuxDiT sun nodes (v5)

Three ComfyUI nodes that replace **LuxDiT Sun (gated)** in the photo → 4K HDRI workflow, plus a click widget.
Ready-to-load workflow: `workflows/photo_to_hdri_4k_v5_api.json` (API format; drag it onto ComfyUI).

Part of ComfyUI-LuxDiT (files `sun_prior.py`, `photo_check.py`, `js/sun_picker.js`).
Needs numpy, OpenCV and torch (already in ComfyUI). The photo check also needs `transformers` ≥ 4.57
(ComfyUI ships 5.x) and downloads Qwen3-VL from Hugging Face on first use (4B ≈ 9 GB).

## What changed and why (measured on 48 blind random Poly Haven images)

| vs current workflow (v3) | standard diffuse | standard mirror | linear diffuse | linear glossy |
|---|---|---|---|---|
| **v5** (this) | −16 % | worse (+0.030) | −0.079 | −0.046 |
| v5, sun off | −16 % | unchanged | −0.011 | +0.005 |

- Indoor and night scenes are left alone (LuxDiT's lighting and added suns made them worse).
- Outdoors, 50 % of LuxDiT's low-frequency lighting is blended in, as a smooth brightness gain that never dims
  any part of the panorama below 0.7× (v5.1; the earlier added correction could turn dark ground into a brown
  smudge or put a white haze on dark forest floors). On 127 blind images v5.1 vs v5: matte −0.010, mirror −0.008,
  diffuse +0.004, linear metric unchanged.
- The sun is added only when the **photo** looks sunny (vision-language model), placed at the panorama's glow,
  carrying 55 % of all light (real suns: about 70 %).
- The standard benchmark's mirror sphere only rewards a sun in exactly the right 1.4° pixel, so it gets worse
  whenever a sun is added; renders with cast shadows (see `docs/results/2026-11-18-v5.md`) are what improves.

## Nodes

### LuxDiT Photo Check (VLM)
`photo` → `p_sunny`, `p_indoor`, `report`. Asks Qwen3-VL about the input photo (forced-choice answers, probabilities
from the logits). `p_indoor` = max(indoor, night). 4B is the validated default (as good as 8B here; ~4 s per photo
on an Apple GPU); 2B is faster.

### LuxDiT Sun (prior)
Inputs: `hdr` (merge output), `ldr_high` (4K LDR), optional `luxdit_hdr`, `p_sunny`, `p_indoor`.
Outputs: `hdr`, `report`, `preview` (tone-mapped, with markers), `sun_json`, `sky_hdr` (no sun disc).

| Control | Meaning |
|---|---|
| `sun_mode` | auto · force sun · no sun · keep generated (leave DiffHDR's sun, only blend) |
| `direction` | auto (panorama glow; LuxDiT's direction when the glow is ambiguous) · click · shadow · angles |
| `points` | written by clicking the preview: click = sun; shift-click base, top, shadow tip = shadow construction |
| `sun_azimuth` / `sun_elevation` | angles mode. Azimuth 0 = centre of the photo, +90 = right, 180 = behind |
| `snap_deg` | click mode: move to the brightest point within this radius (0 = exactly where clicked) |
| `target_share`, `strength_preset`, `sun_ev` | sun strength: share of all light (presets hazy 0.45 / typical 0.65 / clear 0.80), ±stops on the sun-to-sky ratio |
| `sun_radius_deg` | disc radius = shadow softness (default 0.27 = the real sun; same lighting as 0.5, smaller benchmark mirror penalty) |
| `sun_kelvin` | colour temperature; 0 = automatic from elevation |
| `luxdit_blend` / `luxdit_blend_outdoor` | LuxDiT low-frequency blend without / with the photo check saying outdoors (0.25 / 0.75) |
| `threshold`, `detector` | built-in detector when no photo check: v3 (needs `luxdit_hdr`) or v1 |
| `photo_threshold`, `indoor_threshold` | thresholds on `p_sunny` / `p_indoor` (0.5 / 0.5) |

Strength trade-off (measured): `target_share` 0.40 halves the benchmark's mirror penalty and keeps ~80 % of the
linear lighting gain of 0.55; 0.70 (closer to real suns) gains nothing more and costs more on the mirror sphere.

Preview: yellow = the sun placed, white rings a/b/c = other glows in the panorama (click one to use it), grey ring =
the automatic choice when you overrode it, cyan = your clicks.

Workflow tip: queue once, then click the sun on the node's preview and queue again — only this node and the
saves re-run (about a second). Giving any direction implies you want a sun, even in auto mode.

`sun_json` gives direction (also as xyz: right, up, forward), share, irradiance, `lamp_color` and `lamp_strength`
(colour × strength = the sun's irradiance in the HDR's units) and the angular diameter, to rebuild the sun as a
directional lamp next to `sky_hdr` in Blender, Unreal, Houdini, etc.

### LuxDiT Save Text (JSON)
Saves a STRING (e.g. `sun_json`) next to the EXRs as `<prefix>_00001_.json`.

## v5.2: sun halo and photo paste

- **Sun halo.** A bare 0.27° disc looks like a bright dot. Real Poly Haven suns put ~5% of their energy in a glow
  falling off as 1/angle² out to ~12° (measured on 15 real suns); `sun_halo` (default 0.05) adds exactly that.
  Lighting is unchanged (linear metric ±0.000 on 127 blind images; mirror sphere +0.003).
- **LuxDiT Photo Paste.** Puts the original photo back into the generated panorama: SIFT + RANSAC homography between
  the photo and a pinhole view of the panorama's front (both are views from the same point), then a feathered paste
  into the 4K LDR before the HDR merge, so the final HDRI shows the real photo where the camera looked. Skips itself
  (panorama unchanged, reported) when fewer than `min_inliers` points match (8 of 79 blind images: night, snow,
  featureless). Lighting scores unchanged; it is a visual fix. Outputs the panorama, the blend mask and a report.
