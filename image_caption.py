import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import easyocr
import torch
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
TRANSLATION_MODEL = "HaiderSultanArc/t5-small-english-to-urdu"

CAPTION_LOCAL = MODELS_DIR / "blip"
TRANSLATE_LOCAL = MODELS_DIR / "t5_ur"


# ----------------- MODEL DOWNLOAD FUNCTION -----------------
def download(model_name, local_dir, loader, processor=None):
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"{model_name} already exists.")
        return

    print(f"Downloading {model_name}")
    local_dir.mkdir(parents=True, exist_ok=True)

    model = loader(model_name)
    model.save_pretrained(local_dir)

    if processor:
        p = processor(model_name)
        p.save_pretrained(local_dir)

    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.save_pretrained(local_dir)
    except:
        pass

    print(f"{model_name} downloaded.")


download(CAPTION_MODEL, CAPTION_LOCAL, BlipForConditionalGeneration.from_pretrained, BlipProcessor.from_pretrained)
download(TRANSLATION_MODEL, TRANSLATE_LOCAL, AutoModelForSeq2SeqLM.from_pretrained)

# ----------------- LOAD MODELS -----------------
caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_LOCAL).to(DEVICE)
caption_processor = BlipProcessor.from_pretrained(CAPTION_LOCAL)

translator = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATE_LOCAL).to(DEVICE)
translator_tok = AutoTokenizer.from_pretrained(TRANSLATE_LOCAL)

ocr_reader = easyocr.Reader(['en'], gpu=(DEVICE == "cuda"))

# ----------------- UTILITY FUNCTIONS -----------------
def blip_caption(img):
    inputs = caption_processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = caption_model.generate(**inputs, max_length=50)
    return caption_processor.decode(out[0], skip_special_tokens=True)


def extract_ocr(image_path):
    result = ocr_reader.readtext(image_path)
    texts = [r[1] for r in result]
    return texts


def detect_arrows(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=30, maxLineGap=15)
    return [] if lines is None else lines.tolist()


def translate_urdu(text):
    inp = translator_tok.encode(text, return_tensors="pt").to(DEVICE)
    out = translator.generate(inp, max_length=250)
    return translator_tok.decode(out[0], skip_special_tokens=True)


# ----------------- GENERIC SEMANTIC CAPTION -----------------
def semantic_flowchart_caption(scene_caption, ocr_texts, arrows):
    texts = list(set([t.strip() for t in ocr_texts if t.strip()]))

    # Detect numbered steps
    numbered = sorted([t for t in texts if t.replace(" ", "").isdigit()], key=lambda x: int(x))
    labels = [t for t in texts if not t.replace(" ", "").isdigit()]

    scene = scene_caption.capitalize()

    # If structured flow detected
    if numbered or arrows:
        key_elems = ", ".join(labels) if labels else "Key visual elements"

        flow = " → ".join(numbered) if numbered else " → ".join(labels)

        caption_en = (
            f"Scene: {scene}.\n"
            f"Elements: {key_elems}.\n"
            # f"Purpose: Visualizing a structured step-by-step flow.\n"
            # f"For: business explanation, documentation, or presentation use.\n"
            # f"Summary Flow: {flow}."
        )
        return caption_en

    # General semantic caption if no structured flow
    key_elems = ", ".join(labels) if labels else "common visual elements"

    caption_en = (
        f"Scene: {scene}.\n"
        f"Elements: {key_elems}.\n"
        # f"Purpose: Understanding the contents and context of the visual.\n"
        # f"For: general viewers, analysis, or documentation.\n"
        # f"Summary Flow: Scene → Elements → Purpose → Audience."
    )

    return caption_en


# ----------------- IMAGE ANALYSIS -----------------
def analyze_image(path):
    img_pil = Image.open(path).convert("RGB")
    img_cv = cv2.imread(path)

    scene_caption = blip_caption(img_pil)
    ocr_texts = extract_ocr(path)
    arrows = detect_arrows(img_cv)

    final_en = semantic_flowchart_caption(scene_caption, ocr_texts, arrows)
    final_ur = translate_urdu(final_en)

    return final_en, final_ur


# ----------------- CLI INTERFACE -----------------
if __name__ == "__main__":
    print("\n===== UNIVERSAL IMAGE CAPTIONING =====")

    while True:
        img = input("\nEnter image path (or 'exit'): ").strip()
        if img.lower() == "exit":
            break

        if not os.path.exists(img):
            print("❌ Not found.")
            continue

        en, ur = analyze_image(img)

        print("\n--- OUTPUT (English) ---")
        print(en)

        # print("\n--- OUTPUT (Urdu) ---")
        # print(ur)
