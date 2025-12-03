# from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor
# from PIL import Image
#
# # Load model
# model_path = "cnmoro/tiny-image-captioning"
# model = VisionEncoderDecoderModel.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# image_processor = AutoImageProcessor.from_pretrained(model_path)
#
# # ----------------------------
# # Load image from local PC
# # ----------------------------
# image_path = r"C:\Users\Haseeb Ishtiaq\Desktop\Picture1.png"   # ← apna path yahan do
#
# try:
#     image = Image.open(image_path).convert("RGB")
# except Exception as e:
#     print("Image load error:", e)
#     exit()
#
# # Preprocess
# pixel_values = image_processor(image, return_tensors="pt").pixel_values
#
# # Caption Generate
# generated_ids = model.generate(
#     pixel_values,
#     temperature=0.7,
#     top_p=0.9,
#     top_k=50,
#     num_beams=3,
#     max_length=30
# )
#
# generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print("\nCaption:", generated_text)


# from transformers import (
#     VisionEncoderDecoderModel,
#     AutoTokenizer,
#     AutoImageProcessor,
#     pipeline
# )
# from PIL import Image
# import easyocr
#
# # ------------------------------
# # Load models once
# # ------------------------------
# caption_model_path = "cnmoro/tiny-image-captioning"
# caption_model = VisionEncoderDecoderModel.from_pretrained(caption_model_path)
# caption_tokenizer = AutoTokenizer.from_pretrained(caption_model_path)
# caption_processor = AutoImageProcessor.from_pretrained(caption_model_path)
#
# sentiment_model = pipeline("sentiment-analysis")
# ocr_reader = easyocr.Reader(['en'])  # add 'ur' if Urdu text
#
# # ------------------------------
# # Function: Analyze Image
# # ------------------------------
# def analyze_image(image_path):
#     """
#     Takes an image path and returns a dictionary with:
#     - Caption
#     - OCR text
#     - Sentiment
#     - Final combined analysis
#     """
#     # Load image
#     image = Image.open(image_path).convert("RGB")
#
#     # ---- Captioning ----
#     pixel_values = caption_processor(image, return_tensors="pt").pixel_values
#     generated_ids = caption_model.generate(
#         pixel_values,
#         do_sample=True,
#         temperature=0.7,
#         top_p=0.9,
#         top_k=50,
#         max_new_tokens=25
#     )
#     caption = caption_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#
#     # ---- OCR Text ----
#     ocr_result = ocr_reader.readtext(image_path)
#     extracted_text = " ".join([item[1] for item in ocr_result]).strip()
#
#     # ---- Sentiment ----
#     if extracted_text:
#         sent = sentiment_model(extracted_text)[0]
#     else:
#         sent = {"label": "NO_TEXT_FOUND", "score": 0.0}
#
#     # ---- Final Combined Analysis ----
#     if extracted_text:
#         analysis = (
#             f"The image shows: {caption}.\n"
#             f"Detected text: \"{extracted_text}\".\n"
#             f"Sentiment of the text is: {sent['label']} (score {sent['score']:.2f}).\n"
#             f"Overall, the image likely represents a situation where the written text "
#             f"and the visible scene are connected in a {sent['label'].lower()} context."
#         )
#     else:
#         analysis = (
#             f"The image shows: {caption}.\n"
#             f"No text detected inside the image.\n"
#             f"Overall description is based only on visual content."
#         )
#
#     # ---- Return results ----
#     return {
#         "caption": caption,
#         "ocr_text": extracted_text,
#         "sentiment": sent,
#         "analysis": analysis
#     }
#
# # ------------------------------
# # Example Usage
# # ------------------------------
# image_path = r"C:\Users\Haseeb Ishtiaq\Desktop\Picture1.png"
# result = analyze_image(image_path)
#
# print("\n🔹 Caption:", result["caption"])
# print("🔹 OCR Text:", result["ocr_text"])
# print("🔹 Sentiment:", result["sentiment"])
# print("\n🔹 Final Analysis:\n", result["analysis"])

# import os
# from transformers import BlipForConditionalGeneration, BlipProcessor, pipeline
# from PIL import Image
# import easyocr
# import torch
#
# # ============================================================
# # 1) SETTINGS & DEVICE
# # ============================================================
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# # Local model path
# LOCAL_MODEL_PATH = r"C:\local_models\blip"
# MODEL_NAME = "Salesforce/blip-image-captioning-base"
#
# # ============================================================
# # 2) DOWNLOAD OR LOAD BLIP MODEL
# # ============================================================
#
# if not os.path.exists(LOCAL_MODEL_PATH):
#     print(f"[INFO] Local BLIP model not found at {LOCAL_MODEL_PATH}. Downloading...")
#     os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
#     caption_model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
#     caption_processor = BlipProcessor.from_pretrained(MODEL_NAME)
#     caption_model.save_pretrained(LOCAL_MODEL_PATH)
#     caption_processor.save_pretrained(LOCAL_MODEL_PATH)
#     print("[INFO] Model downloaded and saved locally.")
# else:
#     print(f"[INFO] Loading BLIP model from local path: {LOCAL_MODEL_PATH}")
#     caption_model = BlipForConditionalGeneration.from_pretrained(LOCAL_MODEL_PATH)
#     caption_processor = BlipProcessor.from_pretrained(LOCAL_MODEL_PATH)
#
# caption_model = caption_model.to(device)
#
# # ============================================================
# # 3) LOAD OTHER MODELS (OCR + SENTIMENT)
# # ============================================================
#
# reader = easyocr.Reader(['en'])
#
# # For sentiment-analysis, download once and cache
# sentiment_model = pipeline("sentiment-analysis")
#
# # ============================================================
# # 4) SMART CAPTION LOGIC
# # ============================================================
#
# def build_smart_caption(scene_caption, ocr_lines, sentiments):
#     scene_caption = scene_caption.strip().rstrip('.')
#
#     if not ocr_lines:
#         return f"{scene_caption.capitalize()}."
#
#     text_join = " ".join(ocr_lines).lower()
#     positive = sum(1 for s in sentiments if s["label"] == "POSITIVE")
#     negative = sum(1 for s in sentiments if s["label"] == "NEGATIVE")
#
#     # Conversation detection
#     convo_words = ["hey", "hi", "hello", "you", "thank", "good", "ok", "understand"]
#     if any(w in text_join for w in convo_words):
#         if "understand" in text_join:
#             return "A moment where two people are interacting and finally understanding each other better."
#         if negative > positive:
#             return "The image shows a conversation that feels a bit tense or confusing."
#         return "The image shows a friendly conversation happening between two people."
#
#     # Emotional interpretation
#     if positive > negative:
#         return f"{scene_caption.capitalize()}, giving a positive and friendly impression."
#     if negative > positive:
#         return f"{scene_caption.capitalize()}, showing some emotional tension or confusion."
#
#     # General fallback
#     return f"{scene_caption.capitalize()}, with some text visible in the scene."
#
# # ============================================================
# # 5) IMAGE ANALYZER FUNCTION
# # ============================================================
#
# def analyze_image(image_path):
#     if not os.path.exists(image_path):
#         print(f"[ERROR] File not found: {image_path}")
#         return None
#
#     image = Image.open(image_path).convert("RGB")
#
#     # --- BLIP CAPTION ---
#     inputs = caption_processor(images=image, return_tensors="pt").to(device)
#     gen = caption_model.generate(**inputs, max_length=30, num_beams=5, early_stopping=True)
#     scene_caption = caption_processor.decode(gen[0], skip_special_tokens=True)
#
#     # --- OCR ---
#     ocr = reader.readtext(image_path)
#     ocr_lines = [item[1] for item in ocr]
#
#     # --- SENTIMENT ---
#     sentiments = []
#     for line in ocr_lines:
#         s = sentiment_model(line)[0]
#         sentiments.append({
#             "text": line,
#             "label": s["label"],
#             "score": round(s["score"], 3)
#         })
#
#     # --- SMART CAPTION ---
#     final_caption = build_smart_caption(scene_caption, ocr_lines, sentiments)
#
#     return {
#         "scene_caption": scene_caption,
#         "ocr": ocr_lines,
#         "sentiment": sentiments,
#         "final_caption": final_caption
#     }
#
# # ============================================================
# # 6) MAIN LOOP
# # ============================================================
#
# print("\n[INFO] Image Captioning + OCR + Sentiment Script")
# print("[INFO] Type 'exit' to quit the program.\n")
#
# while True:
#     image_path = input("Enter image path: ").strip()
#     if image_path.lower() == "exit":
#         print("[INFO] Exiting program...")
#         break
#
#     result = analyze_image(image_path)
#     if result is None:
#         continue
#
#     print("\n================ RESULT ================\n")
#     print("🔹 Final Smart Caption:", result["final_caption"])
#     print("🔹 OCR Text:", result["ocr"])
#     print("🔹 Sentiments:", result["sentiment"])
#     print("🔹 BLIP Raw Caption:", result["scene_caption"])
#     print("\n========================================\n")


import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import easyocr
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor, AutoModelForSeq2SeqLM, AutoTokenizer

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
TRANSLATION_MODEL_NAME = "HaiderSultanArc/t5-small-english-to-urdu"

CAPTION_LOCAL = MODELS_DIR / "blip_caption"
TRANSLATE_LOCAL = MODELS_DIR / "t5_en_ur"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ------------------------------------------------------------
# Helpers: download-if-missing
# ------------------------------------------------------------
def download_if_missing_hf(model_name: str, local_dir: Path, loader_fn, processor_fn=None):
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"Model already present at {local_dir}")
        return

    print(f"Downloading {model_name} into {local_dir} ...")
    local_dir.mkdir(parents=True, exist_ok=True)

    model = loader_fn(model_name)
    model.save_pretrained(local_dir)

    if processor_fn:
        processor = processor_fn(model_name)
        processor.save_pretrained(local_dir)

    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.save_pretrained(local_dir)
    except:
        pass

    print("Downloaded & saved.")

# ------------------------------------------------------------
# Download models if missing
# ------------------------------------------------------------
download_if_missing_hf(CAPTION_MODEL_NAME, CAPTION_LOCAL, BlipForConditionalGeneration.from_pretrained, BlipProcessor.from_pretrained)
download_if_missing_hf(TRANSLATION_MODEL_NAME, TRANSLATE_LOCAL, AutoModelForSeq2SeqLM.from_pretrained)

print("Loading models...")
caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_LOCAL).to(DEVICE)
caption_processor = BlipProcessor.from_pretrained(CAPTION_LOCAL)

translator = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATE_LOCAL).to(DEVICE)
translator_tokenizer = AutoTokenizer.from_pretrained(TRANSLATE_LOCAL)

# EasyOCR for English + Urdu
ocr_reader = easyocr.Reader(['en', 'ur'], gpu=(DEVICE == "cuda"))

# ------------------------------------------------------------
# Image Helpers
# ------------------------------------------------------------
def load_image(path):
    return Image.open(path).convert("RGB")

# ------------------------------------------------------------
# Caption Generator
# ------------------------------------------------------------
def generate_caption(img):
    inputs = caption_processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = caption_model.generate(**inputs, max_length=30)
    return caption_processor.decode(out[0], skip_special_tokens=True)

# ------------------------------------------------------------
# OCR
# ------------------------------------------------------------
def extract_ocr(path):
    results = ocr_reader.readtext(path, detail=True)
    out = []
    for bbox, text, conf in results:
        out.append({"bbox": bbox, "text": text.strip(), "conf": float(conf)})
    return out

# ------------------------------------------------------------
# Arrow Detection
# ------------------------------------------------------------
def detect_arrows(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
    arrows = []
    if lines is None:
        return arrows
    for l in lines:
        x1, y1, x2, y2 = l[0]
        arrows.append({"line": ((x1, y1), (x2, y2))})
    return arrows

# ------------------------------------------------------------
# Arrow → OCR Relations
# ------------------------------------------------------------
def get_relations(ocr_items, arrows):
    def center(bbox):
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return (sum(xs)/len(xs), sum(ys)/len(ys))
    centers = [center(o["bbox"]) for o in ocr_items]
    relations = []
    for arrow in arrows:
        (x1, y1), (x2, y2) = arrow["line"]
        tail = (x1, y1)
        head = (x2, y2)
        def nearest(pt):
            if not centers:
                return None
            d = [((pt[0]-cx)**2 + (pt[1]-cy)**2, idx) for idx, (cx, cy) in enumerate(centers)]
            d.sort()
            return ocr_items[d[0][1]]["text"]
        t_text = nearest(tail)
        h_text = nearest(head)
        if t_text and h_text and t_text != h_text:
            relations.append(f"Relation: '{t_text}' → '{h_text}'")
    return relations

# ------------------------------------------------------------
# Translate EN -> UR
# ------------------------------------------------------------
def translate_to_urdu(text):
    inputs = translator_tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = translator.generate(inputs, max_length=200, num_beams=4)
    return translator_tokenizer.decode(out[0], skip_special_tokens=True)

# ------------------------------------------------------------
# Main Pipeline
# ------------------------------------------------------------
def analyze_image(image_path):
    img_pil = load_image(image_path)
    img_cv = cv2.imread(image_path)

    caption_en = generate_caption(img_pil)
    ocr_items = extract_ocr(image_path)
    arrows = detect_arrows(img_cv)
    relations = get_relations(ocr_items, arrows)

    ocr_texts = [o["text"] for o in ocr_items]
    ocr_summary = ""
    if ocr_texts:
        ocr_summary = " Text found: " + ", ".join(list(set(ocr_texts))) + "."

    relation_summary = ""
    if relations:
        relation_summary = " " + " ".join(relations)

    final_en = caption_en + ocr_summary + relation_summary
    final_ur = translate_to_urdu(final_en)

    return {
        "caption_en": caption_en,
        "ocr": ocr_items,
        "relations": relations,
        "final_en": final_en,
        "final_ur": final_ur,
    }

# ------------------------------------------------------------
# Script Entry — Ask for Image Path
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\n===== IMAGE ANALYSIS PIPELINE =====")
    image_path = input("Enter image path: ").strip()
    if not os.path.exists(image_path):
        print("❌ ERROR: Image not found. Check the path.")
        sys.exit(1)

    result = analyze_image(image_path)

    print("\n----- RESULTS -----")
    print("Caption (English):", result["caption_en"])
    print("\nOCR Texts:", [o["text"] for o in result["ocr"]])
    print("\nRelations:")
    for r in result["relations"]:
        print(" -", r)

    print("\nFINAL ENGLISH:\n", result["final_en"])
    print("\nFINAL URDU:\n", result["final_ur"])
