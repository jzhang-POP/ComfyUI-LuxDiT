"""LuxDiT Photo Check — asks a vision-language model (Qwen3-VL) about the INPUT PHOTO:
is it indoors, and is the scene lit by direct sunlight? Feeds LuxDiT Sun (prior) (p_sunny, p_indoor).

On the random Poly Haven benchmark (72 photos), "direct sunlight?" reached AUC 0.88 vs 0.80 for the sky-feature
detector of the sun node, and indoor/outdoor 92% correct. Probabilities come from the next-token logits of
forced-choice answers. Weights download from Hugging Face on first use (8B: ~17 GB, bf16)."""
import numpy as np

QUESTIONS = {
    "indoor": ("Is this photo taken indoors or outdoors? Answer with exactly one word: indoors or outdoors.",
               ["indoors", "outdoors"]),
    "night": ("Was this photo taken at night or in deep twilight, with a dark sky and the scene lit mostly by "
              "artificial lights or moonlight? Answer with exactly one word: yes or no.", ["yes", "no"]),
    "sunny": ("Is this scene lit by direct sunlight right now, i.e. a sunny scene with hard-edged shadows cast by the "
              "sun? Overcast sky, open shade, twilight after sunset, night, or artificial light count as no. "
              "Answer with exactly one word: yes or no.", ["yes", "no"]),
    "sunny2": ("Look at the shadows and highlights in this photo. Is the sun shining directly on the scene (sunlight "
               "reaching the ground or objects), rather than an overcast, shaded, twilight, night or artificially "
               "lit scene? Answer with exactly one word: yes or no.", ["yes", "no"]),
    "shadows": ("Are there sharp, well-defined shadows cast by direct sunlight anywhere in this photo? "
                "Answer with exactly one word: yes or no.", ["yes", "no"]),
}
MODELS = ["Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3-VL-2B-Instruct"]
MAX_PIXELS = 1_000_000  # the benchmark photos were 720x480; big photos only cost time and memory
_CACHE = {}


class Asker:
    def __init__(self, model_id):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.dev = dev
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16 if dev != "cpu" else torch.float32).eval()
        tok = self.proc.tokenizer
        self.first = {w: tok.encode(w, add_special_tokens=False)[0] for q in QUESTIONS.values() for w in q[1]}

    def to_device(self):
        self.model.to(self.dev)

    def offload(self):  # free GPU memory for LuxDiT / DiffHDR between runs
        self.model.to("cpu")
        if self.dev == "cuda":
            self.torch.cuda.empty_cache()
        elif self.dev == "mps":
            self.torch.mps.empty_cache()

    def ask(self, image, key):
        q, opts = QUESTIONS[key]
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": q}]}]
        inp = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                            return_tensors="pt").to(self.dev)
        with self.torch.no_grad():
            logits = self.model(**inp).logits[0, -1].float()
        z = self.torch.stack([self.torch.log_softmax(logits, -1)[self.first[o]] for o in opts])
        return {o: float(x) for o, x in zip(opts, self.torch.softmax(z, 0).cpu().numpy())}


class LuxDiTPhotoCheck:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "photo": ("IMAGE", {"tooltip": "The input photo (the same image fed to the panorama and LuxDiT)."}),
            "model": (MODELS, {"default": MODELS[0], "tooltip": "Qwen3-VL size. 4B is the validated default (as good as "
                               "8B on the benchmark); 2B is faster. The model is moved off the GPU after each run."}),
        }}

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("p_sunny", "p_indoor", "report")  # p_indoor = indoor OR night: where LuxDiT's lighting hurts
    FUNCTION = "run"
    CATEGORY = "LuxDiT/HDR tools"

    def run(self, photo, model, asker=None, _factory=None):
        from PIL import Image
        if asker is None:
            if model not in _CACHE:
                _CACHE.clear()  # keep one model in memory
                _CACHE[model] = (_factory or Asker)(model)
            asker = _CACHE[model]
        img = Image.fromarray((np.clip(photo[0, ..., :3].float().cpu().numpy(), 0, 1) * 255).astype(np.uint8))
        if img.width * img.height > MAX_PIXELS:
            k = (MAX_PIXELS / (img.width * img.height)) ** 0.5
            img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)
        if hasattr(asker, "to_device"):
            asker.to_device()
        try:
            a = {k: asker.ask(img, k) for k in QUESTIONS}
        finally:
            if hasattr(asker, "offload"):
                asker.offload()
        p_sun = float(np.mean([a["sunny"]["yes"], a["sunny2"]["yes"], a["shadows"]["yes"]]))
        p_in = float(max(a["indoor"]["indoors"], a["night"]["yes"]))
        text = (f"photo check ({model.split('/')[-1]}): sunny {p_sun:.2f} | indoor {a['indoor']['indoors']:.2f} | "
                f"night {a['night']['yes']:.2f}")
        if photo.shape[0] > 1:
            text += f" | batch of {photo.shape[0]}: only the first photo was checked"
        print("[LuxDiT photo check] " + text)
        return (p_sun, p_in, text)


NODE_CLASS_MAPPINGS = {"LuxDiTPhotoCheck": LuxDiTPhotoCheck}
NODE_DISPLAY_NAME_MAPPINGS = {"LuxDiTPhotoCheck": "LuxDiT Photo Check (VLM)"}
