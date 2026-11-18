// LuxDiT Sun (prior): shows the node's panorama preview and turns clicks on it into the `points` widget.
//   click        -> one point, direction = click, sun_mode = force sun
//   shift-click  -> add a point (up to 3), direction = shadow when there are 2 or 3
// Only this node (and what follows it) re-runs when you queue again, so placing the sun takes about a second.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "LuxDiTSunPrior";

function widget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function setWidget(node, name, value) {
  const w = widget(node, name);
  if (!w || w.value === value) return;
  w.value = value;
  w.callback?.(value);
}

function readPoints(node) {
  try {
    const v = JSON.parse(widget(node, "points")?.value || "[]");
    return Array.isArray(v) ? v.filter((p) => Array.isArray(p) && p.length === 2) : [];
  } catch {
    return [];
  }
}

app.registerExtension({
  name: "luxdit.sun_picker",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const node = this;

      const wrap = document.createElement("div");
      wrap.className = "luxdit-sun-picker";
      wrap.style.cssText = "position:relative;width:100%;font:11px sans-serif;color:#bbb;user-select:none;";
      const frame = document.createElement("div");
      frame.style.cssText = "position:relative;width:100%;aspect-ratio:2/1;background:#222;border-radius:4px;overflow:hidden;cursor:crosshair;";
      const img = document.createElement("img");
      img.style.cssText = "position:absolute;inset:0;width:100%;height:100%;display:none;pointer-events:none;";
      const empty = document.createElement("div");
      empty.textContent = "Queue once: the panorama preview appears here. Then click the sun.";
      empty.style.cssText = "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;padding:8px;";
      const dots = document.createElement("div");
      dots.style.cssText = "position:absolute;inset:0;pointer-events:none;";
      frame.append(img, empty, dots);
      const bar = document.createElement("div");
      bar.style.cssText = "display:flex;gap:8px;align-items:center;margin-top:3px;";
      const hint = document.createElement("span");
      hint.textContent = "click: sun · shift-click: shadow (base, top, tip)";
      hint.style.flex = "1";
      const clear = document.createElement("button");
      clear.textContent = "clear";
      clear.style.cssText = "font:11px sans-serif;padding:1px 6px;";
      bar.append(hint, clear);
      wrap.append(frame, bar);

      const draw = () => {
        dots.replaceChildren();
        readPoints(node).forEach(([u, v], i) => {
          const d = document.createElement("div");
          d.style.cssText = `position:absolute;left:${u * 100}%;top:${v * 100}%;transform:translate(-50%,-50%);` +
            "width:12px;height:12px;border-radius:50%;border:2px solid #00dcff;box-shadow:0 0 0 1px #000;";
          const n = document.createElement("span");
          n.textContent = String(i + 1);
          n.style.cssText = "position:absolute;left:12px;top:-12px;color:#00dcff;font:bold 10px sans-serif;text-shadow:0 0 2px #000;";
          d.append(n);
          dots.append(d);
        });
      };
      node._luxditSun = { img, empty, draw };

      frame.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        const b = frame.getBoundingClientRect();
        const u = Math.min(1, Math.max(0, (e.clientX - b.left) / b.width));
        const v = Math.min(1, Math.max(0, (e.clientY - b.top) / b.height));
        let pts = e.shiftKey ? readPoints(node) : [];
        pts.push([+u.toFixed(4), +v.toFixed(4)]);
        pts = pts.slice(-3);
        setWidget(node, "points", JSON.stringify(pts));
        setWidget(node, "direction", pts.length > 1 ? "shadow" : "click");
        setWidget(node, "sun_mode", "force sun");
        draw();
        node.setDirtyCanvas?.(true, true);
        e.preventDefault();
        e.stopPropagation();
      });
      clear.addEventListener("click", (e) => {
        setWidget(node, "points", "");
        setWidget(node, "direction", "auto");
        setWidget(node, "sun_mode", "auto");
        draw();
        e.stopPropagation();
      });

      const baseHeight = node.computeSize()[1];
      const pickerHeight = () => Math.max(80, ((node.size?.[0] ?? 380) - 20) / 2 + 30);
      const w = node.addDOMWidget("sun_picker", "luxdit_sun_picker", wrap, {
        serialize: false, hideOnZoom: false, getMinHeight: pickerHeight, getMaxHeight: pickerHeight, getHeight: pickerHeight,
      });
      const pw = widget(node, "points");
      if (pw) {
        const cb = pw.callback;
        pw.callback = function () { const x = cb?.apply(this, arguments); draw(); return x; };
      }
      setTimeout(draw, 0);
      const width = Math.max(node.size[0], 380);
      node.setSize?.([width, Math.max(node.computeSize()[1], baseHeight + ((width - 20) / 2 + 30) + 12)]);
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const r = onExecuted?.apply(this, arguments);
      const im = message?.sun_preview?.[0];
      const s = this._luxditSun;
      if (im && s) {
        const q = new URLSearchParams({ filename: im.filename, type: im.type, subfolder: im.subfolder || "" });
        s.img.src = api.apiURL(`/view?${q}&t=${Date.now()}`);
        s.img.style.display = "block";
        s.empty.style.display = "none";
        s.draw();
      }
      return r;
    };
  },
});
