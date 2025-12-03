# # from sentence_transformers import SentenceTransformer
# #
# # print("🔄 Downloading clean model...")
# # model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder="models")
# #
# # print("✅ Model loaded successfully!")
# #
# # # Test to confirm
# # sentences = ["This is a test", "Embeddings are ready!"]
# # embeddings = model.encode(sentences)
# # print("✅ Embeddings shape:", embeddings.shape)
#
# """
# download_summary_model.py
# --------------------------
# This script downloads and verifies the BART summarization model
# (`facebook/bart-large-cnn`) used for text/document summarization.
# """
#
# from transformers import pipeline
# import os
#
# MODEL_NAME = "facebook/bart-large-cnn"
# CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
#
# print("🔍 Checking for model in local cache...")
# model_path = os.path.join(CACHE_DIR, f"models--{MODEL_NAME.replace('/', '--')}")
#
# if os.path.exists(model_path):
#     print(f"✅ Model already exists locally at:\n{model_path}")
# else:
#     print("⬇️ Model not found locally — downloading from Hugging Face...")
#     summarizer = pipeline("summarization", model=MODEL_NAME)
#     print("✅ Model downloaded and cached successfully!")
#
# # === Test summary generation ===
# print("\n🧠 Testing summarization model...")
# summarizer = pipeline("summarization", model=MODEL_NAME)
# text = (
#     "Artificial intelligence is rapidly transforming industries across the world. "
#     "It is enabling automation, improving decision-making, and creating new opportunities "
#     "for innovation in fields such as healthcare, education, and finance."
# )
# summary = summarizer(text, max_length=50, min_length=20, do_sample=False)[0]['summary_text']
#
# print("\n📄 Example summary:")
# print(summary)
#
# print("\n✅ Model is ready for use in your main project!")

"""
summarize_and_speak.py
----------------------
This script:
1. Loads the local `facebook/bart-large-cnn` model for summarization.
2. Generates a text summary.
3. Converts that summary into speech (audio).
"""

"""
summary_to_audio.py
--------------------
This script:
1. Generates a text summary using facebook/bart-large-cnn
2. Converts the summary into speech using gTTS (Google Text-to-Speech)
3. Saves the output as summary_audio.mp3
"""

# from transformers import pipeline
# import pyttsx3
# import os
#
# # === STEP 1: Generate Summary ===
# print("🔄 Loading summarization model...")
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
#
# # Example text (you can replace this with your own input)
# text = """
# Artificial intelligence is revolutionizing industries around the world.
# It enables automation, enhances decision-making, and creates new possibilities
# in healthcare, education, and financial systems. However, ethical considerations
# and data privacy concerns continue to grow as AI becomes more integrated into society.
# """
#
# print("🧠 Generating summary...")
# summary = summarizer(text, max_length=80, min_length=30, do_sample=False)[0]['summary_text']
# print("\n📄 Generated Summary:")
# print(summary)
#
# # === STEP 2: Convert Summary to Audio (Offline using pyttsx3) ===
# print("\n🎤 Converting summary to speech (offline)...")
# os.makedirs("audio_output", exist_ok=True)
# output_path = "audio_output/summary_audio.wav"
#
# # Initialize pyttsx3 engine
# engine = pyttsx3.init()
#
# # Optional: Set speaking rate and volume
# engine.setProperty('rate', 160)   # Speed of speech (default ~200)
# engine.setProperty('volume', 1.0) # Volume (0.0 to 1.0)
#
# # Optional: Change voice (male/female depending on system voices)
# voices = engine.getProperty('voices')
# if voices:
#     engine.setProperty('voice', voices[0].id)  # Use voices[1] for female on Windows
#
# # Save the summary to an audio file
# engine.save_to_file(summary, output_path)
# engine.runAndWait()
#
# print(f"✅ Offline audio summary saved at: {output_path}")


# ppt reader just for text and image
# from pptx import Presentation
# from PIL import Image
# import pytesseract
# from transformers import pipeline
# import pyttsx3
# import io
# import os
#
# # --- Setup Tesseract path ---
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#
# # --- Load summarization model ---
# print("🔄 Loading summarization model (may take a moment)...")
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# print("✅ Model loaded.\n")
#
# # --- Extract text and OCR from PPTX ---
# def extract_text_and_ocr(pptx_path):
#     normal_text = ""
#     ocr_text = ""
#
#     prs = Presentation(pptx_path)
#     for slide_idx, slide in enumerate(prs.slides, start=1):
#         slide_normal = ""
#         slide_ocr = ""
#         for shape in slide.shapes:
#             if hasattr(shape, "text") and shape.text.strip():
#                 slide_normal += shape.text + "\n"
#
#             if shape.shape_type == 13:  # Picture
#                 img = Image.open(io.BytesIO(shape.image.blob))
#                 ocr_result = pytesseract.image_to_string(img)
#                 if ocr_result.strip():
#                     slide_ocr += ocr_result + "\n"
#
#         if slide_normal.strip():
#             normal_text += f"[Slide {slide_idx}]\n{slide_normal}\n"
#         if slide_ocr.strip():
#             ocr_text += f"[Slide {slide_idx}]\n{slide_ocr}\n"
#
#     return normal_text.strip(), ocr_text.strip()
#
#
# # --- Chunked summarization ---
# def summarize_in_chunks(text, chunk_size=1500):
#     """Split text into chunks to avoid token limit and summarize each."""
#     summaries = []
#     for i in range(0, len(text), chunk_size):
#         chunk = text[i:i+chunk_size]
#         summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
#         summaries.append(summary)
#     return " ".join(summaries)
#
#
# def summarize(text):
#     if not text.strip():
#         return "No content found."
#     return summarize_in_chunks(text, chunk_size=1500)
#
#
# # --- Generate offline audio using pyttsx3 ---
# def generate_audio(text, output_path):
#     engine = pyttsx3.init()
#     engine.setProperty('rate', 150)  # Speed
#     engine.setProperty('volume', 1.0)
#     engine.save_to_file(text, output_path)
#     engine.runAndWait()
#     print(f"✅ Audio saved at: {output_path}")
#
#
# # --- Process PPTX ---
# def process_pptx(file_path):
#     normal_text, ocr_text = extract_text_and_ocr(file_path)
#
#     print("\n📝 Summarizing normal text...")
#     normal_summary = summarize(normal_text)
#     print("📝 Summarizing OCR (image) text...")
#     ocr_summary = summarize(ocr_text)
#
#     # --- Save combined summary to text file ---
#     output_txt = "summary_with_sections.txt"
#     with open(output_txt, "w", encoding="utf-8") as f:
#         f.write("===== Normal Slide Text Summary =====\n")
#         f.write(normal_summary + "\n\n")
#         f.write("===== Image Text (OCR) Summary =====\n")
#         f.write(ocr_summary + "\n")
#     print(f"✅ Summary saved at {output_txt}")
#
#     # --- Combine text for single audio ---
#     combined_text = f"Normal Slide Text Summary: {normal_summary}. Image Text Summary: {ocr_summary}."
#     output_audio = "summary_audio.mp3"
#     generate_audio(combined_text, output_audio)
#
#
# # --- Main ---
# if __name__ == "__main__":
#     pptx_file = input("Enter PPTX file path: ").strip('"')
#     if os.path.exists(pptx_file):
#         process_pptx(pptx_file)
#     else:
#         print("❌ File not found. Please check the path.")

# with urdu and english ppt reader and summary generator
# from pptx import Presentation
# from PIL import Image
# import pytesseract
# from transformers import pipeline
# from deep_translator import GoogleTranslator
# import pyttsx3
# import io
# import os
# from gtts import gTTS
# # import requests
#
# # --- Setup Tesseract path ---
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#
# # --- Load summarization model ---
# print("🔄 Loading summarization model (may take a moment)...")
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# print("✅ Model loaded.\n")
#
#
# # --- Extract text + OCR from PPTX ---
# def extract_text_and_ocr(pptx_path):
#     normal_text = ""
#     ocr_text = ""
#
#     prs = Presentation(pptx_path)
#     for slide_idx, slide in enumerate(prs.slides, start=1):
#         slide_normal = ""
#         slide_ocr = ""
#         for shape in slide.shapes:
#             if hasattr(shape, "text") and shape.text.strip():
#                 slide_normal += shape.text + "\n"
#
#             if shape.shape_type == 13:  # Picture type
#                 try:
#                     img = Image.open(io.BytesIO(shape.image.blob))
#                     ocr_result = pytesseract.image_to_string(img)
#                     if ocr_result.strip():
#                         slide_ocr += ocr_result + "\n"
#                 except Exception as e:
#                     print(f"⚠️ OCR failed on slide {slide_idx}: {e}")
#
#         if slide_normal.strip():
#             normal_text += f"[Slide {slide_idx}]\n{slide_normal}\n"
#         if slide_ocr.strip():
#             ocr_text += f"[Slide {slide_idx}]\n{ocr_text}\n"
#
#     return normal_text.strip(), ocr_text.strip()
#
#
# # --- Chunked summarization ---
# def summarize_in_chunks(text, chunk_size=1500):
#     summaries = []
#     for i in range(0, len(text), chunk_size):
#         chunk = text[i:i + chunk_size]
#         summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
#         summaries.append(summary)
#     return " ".join(summaries)
#
#
# def summarize(text):
#     if not text.strip():
#         return "No content found."
#     return summarize_in_chunks(text, chunk_size=1500)
#
#
# # --- Translate English → Urdu ---
# def translate_to_urdu(text):
#     try:
#         # Split long text for translation (Google Translator has limits)
#         if len(text) > 4000:
#             chunks = [text[i:i + 4000] for i in range(0, len(text), 4000)]
#             translated_chunks = []
#             for chunk in chunks:
#                 translated = GoogleTranslator(source='auto', target='ur').translate(chunk)
#                 translated_chunks.append(translated)
#             return " ".join(translated_chunks)
#         else:
#             return GoogleTranslator(source='auto', target='ur').translate(text)
#     except Exception as e:
#         print(f"⚠️ Translation failed: {e}")
#         return text
#
#
# # --- Generate English Audio using pyttsx3 ---
# def generate_english_audio(text, output_path):
#     try:
#         engine = pyttsx3.init()
#         engine.setProperty('rate', 150)
#         engine.setProperty('volume', 1.0)
#
#         # Set English voice
#         voices = engine.getProperty('voices')
#         for voice in voices:
#             if 'english' in voice.name.lower() or 'en_' in voice.id.lower():
#                 engine.setProperty('voice', voice.id)
#                 break
#
#         engine.save_to_file(text, output_path)
#         engine.runAndWait()
#         print(f"✅ English audio saved: {output_path}")
#     except Exception as e:
#         print(f"❌ English audio generation failed: {e}")
#
#
# # --- Generate Urdu Audio using gTTS (Online) ---
# def generate_urdu_audio(text, output_path):
#     try:
#         print("🌐 Generating Urdu audio using gTTS...")
#
#         # Clean text for TTS
#         clean_text = text.replace('\n', ' ').replace('  ', ' ').strip()
#
#         # Use gTTS for Urdu (online service)
#         tts = gTTS(text=clean_text, lang='ur', slow=False)
#         tts.save(output_path)
#         print(f"✅ Urdu audio saved: {output_path}")
#
#     except Exception as e:
#         print(f"❌ Urdu audio generation failed: {e}")
#         print("💡 Trying alternative method...")
#         generate_urdu_audio_alternative(text, output_path)
#
#
# # --- Alternative Urdu Audio Method ---
# def generate_urdu_audio_alternative(text, output_path):
#     try:
#         # Using pyttsx3 with available voices as fallback
#         engine = pyttsx3.init()
#         engine.setProperty('rate', 120)
#         engine.setProperty('volume', 1.0)
#
#         # Try to find any compatible voice
#         voices = engine.getProperty('voices')
#         if voices:
#             engine.setProperty('voice', voices[0].id)
#
#         engine.save_to_file(text, output_path)
#         engine.runAndWait()
#         print(f"✅ Urdu audio (alternative) saved: {output_path}")
#     except Exception as e:
#         print(f"❌ All Urdu audio methods failed: {e}")
#
#
# # --- Check Urdu Text ---
# def check_urdu_text(text):
#     """Check if text contains Urdu characters"""
#     urdu_range = range(0x0600, 0x06FF)  # Arabic script range (includes Urdu)
#     return any(ord(char) in urdu_range for char in text)
#
#
# # --- Process PPTX file ---
# def process_pptx(file_path):
#     os.makedirs("output", exist_ok=True)
#     base_name = os.path.splitext(os.path.basename(file_path))[0]
#
#     # Extract text
#     print("\n📂 Extracting text and OCR from PPT...")
#     normal_text, ocr_text = extract_text_and_ocr(file_path)
#
#     # English Summary
#     print("\n🧠 Generating English summary...")
#     normal_summary = summarize(normal_text) if normal_text.strip() else "No text content found."
#     ocr_summary = summarize(ocr_text) if ocr_text.strip() else "No image text content found."
#     combined_summary_en = f"Normal Slide Text Summary:\n{normal_summary}\n\nImage Text Summary:\n{ocr_summary}"
#
#     # Save English text
#     english_summary_path = f"output/{base_name}_english_summary.txt"
#     with open(english_summary_path, "w", encoding="utf-8") as f:
#         f.write(combined_summary_en)
#     print(f"✅ English summary saved at: {english_summary_path}")
#
#     # English audio
#     english_audio_path = f"output/{base_name}_english_audio.mp3"
#     generate_english_audio(combined_summary_en, english_audio_path)
#
#     # Urdu Translation
#     print("\n🌐 Translating summary to Urdu...")
#     urdu_summary = translate_to_urdu(combined_summary_en)
#
#     # Check if translation worked
#     if check_urdu_text(urdu_summary):
#         print("✅ Urdu translation contains Urdu characters")
#     else:
#         print("⚠️ Urdu translation might not have worked properly")
#
#     urdu_summary_path = f"output/{base_name}_urdu_summary.txt"
#     with open(urdu_summary_path, "w", encoding="utf-8") as f:
#         f.write(urdu_summary)
#     print(f"✅ Urdu summary saved at: {urdu_summary_path}")
#
#     # Urdu audio - Using gTTS for proper Urdu support
#     urdu_audio_path = f"output/{base_name}_urdu_audio.mp3"
#     generate_urdu_audio(urdu_summary, urdu_audio_path)
#
#     print("\n🎯 All tasks completed successfully!")
#     print(f"📄 English summary: {english_summary_path}")
#     print(f"🔊 English audio: {english_audio_path}")
#     print(f"📄 Urdu summary: {urdu_summary_path}")
#     print(f"🔊 Urdu audio: {urdu_audio_path}")
#
#
# # --- Main ---
# if __name__ == "__main__":
#     file_path = input("👉 Enter PPTX file path: ").strip()
#     file_path = file_path.replace("\u202a", "").replace("\u202b", "")  # remove hidden Unicode chars
#
#     if os.path.exists(file_path):
#         process_pptx(file_path)
#     else:
#         print(f"❌ File not found. Please check the path:\n{file_path}")


# equation handling
# import re
# import os
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#
# # ----------------------------
# # CONFIG
# # ----------------------------
# MODEL_NAME = "microsoft/phi-2"   # ~1.3GB, best small model
# MODEL_DIR = "models/phi2"
#
# # ----------------------------
# # LOAD MODEL (Auto-download if missing)
# # ----------------------------
# def load_phi2():
#     print("🔄 Checking model...")
#
#     if not os.path.exists(MODEL_DIR):
#         print("📥 Model not found, downloading...")
#         os.makedirs(MODEL_DIR, exist_ok=True)
#
#     print("🔄 Loading Phi-2 model (offline after first download)...")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         cache_dir=MODEL_DIR,
#         torch_dtype="auto",
#         device_map="auto"
#     )
#
#     print("✅ Phi-2 Loaded Successfully!")
#     return pipeline("text-generation", model=model, tokenizer=tokenizer)
#
#
# phi2 = load_phi2()
#
#
# # ----------------------------
# # FUNCTION: Equation Explanation
# # ----------------------------
# def explain_equations_in_text(text: str) -> str:
#     eq_patterns = [
#         r'([A-Za-z0-9]+ ?= ?[A-Za-z0-9\+\-\*/\^\(\)±√]+)',
#         r'(\\[a-z]+{[^}]+})',
#         r'([A-Za-z]+ ?\([A-Za-z0-9, ]*\) ?= ?[A-Za-z0-9\+\-\*/\^\(\)]+)',
#     ]
#
#     for pat in eq_patterns:
#         matches = re.findall(pat, text)
#         for eq in matches:
#             prompt = (
#                 f"Explain this equation clearly:\n\n"
#                 f"Equation: {eq}\n"
#                 f"Explain:\n"
#                 f"- What does it mean?\n"
#                 f"- Who introduced it?\n"
#                 f"- Why is it important?\n"
#                 f"- Where is it used?\n"
#                 f"- Give a simple real-life example.\n"
#             )
#
#             try:
#                 response = phi2(prompt, max_length=250, temperature=0.3)[0]["generated_text"]
#             except:
#                 response = f"[Failed to explain: {eq}]"
#
#             text = text.replace(eq, f"[Equation Explanation → {response}]")
#
#     return text
#
#
# # ----------------------------
# # TEST AREA
# # ----------------------------
# if __name__ == "__main__":
#     sample = """
#     Newton's second law is F = ma.
#     The quadratic formula is x = (-b ± √(b^2 - 4ac)) / 2a.
#     """
#
#     print("🔍 Output:")
#     print(explain_equations_in_text(sample))
#

# equatin handling
# import os
# import re
# import requests
# from gpt4all import GPT4All
#
# # =====================================
# # CONFIG
# # =====================================
#
# MODEL_NAME = "TinyLlama-1.1B-Chat-v0.3-GGUF"
# MODEL_FILE = "tinyllama.gguf"
# MODEL_PATH = f"models/{MODEL_FILE}"
#
# MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_0.gguf"
#
#
# # =====================================
# # DOWNLOAD IF NOT EXIST
# # =====================================
#
# def ensure_model_exists():
#     if not os.path.exists("models"):
#         os.makedirs("models")
#
#     if not os.path.isfile(MODEL_PATH):
#         print("📥 Model not found — downloading TinyLlama (350MB)...")
#         response = requests.get(MODEL_URL, stream=True)
#
#         with open(MODEL_PATH, "wb") as f:
#             for chunk in response.iter_content(1024 * 1024):
#                 f.write(chunk)
#
#         print("✅ TinyLlama downloaded!")
#
#     else:
#         print("✅ Model already exists — loading offline...")
#
#
# # =====================================
# # LOAD MODEL
# # =====================================
#
# ensure_model_exists()
# print("🔄 Loading TinyLlama model...")
# model = GPT4All(model_name=MODEL_FILE, model_path="models")
# print("✅ Model Loaded!")
#
#
# # =====================================
# # EQUATION EXPLAIN FUNCTION
# # =====================================
#
# def explain_equations_in_text(text: str) -> str:
#     """
#     Detect and explain equations using TinyLlama.
#     """
#
#     equation_patterns = [
#         r'([A-Za-z0-9]+ ?= ?[A-Za-z0-9\+\-\*/\^\(\)±√]+)',  # e.g., F = ma
#         r'(\\[a-z]+{[^}]+})',  # LaTeX \frac{a}{b}
#         r'([A-Za-z]+ ?\([A-Za-z0-9, ]*\) ?= ?[A-Za-z0-9\+\-\*/\^\(\)]+)',  # f(x)=x^2
#     ]
#
#     for pat in equation_patterns:
#         matches = re.findall(pat, text)
#         for eq in matches:
#
#             print(f"🔍 Explaining: {eq}")
#
#             prompt = f"""
# You are a math expert. Explain the following equation in simple words:
# Equation: {eq}
#
# Explain:
# 1. What does this equation mean?
# 2. Why is it used?
# 3. Where is it used in real life?
# 4. Who introduced it (if known)?
# Keep it short and easy.
# """
#
#             try:
#                 explanation = model.generate(prompt, max_tokens=180).strip()
#             except:
#                 explanation = f"[Failed to explain equation: {eq}]"
#
#             text = text.replace(eq, f"[Equation Explanation]: {explanation}")
#
#     return text
#
#
# # =====================================
# # TEST
# # =====================================
#
# if __name__ == "__main__":
#     sample = """
#     Newton's second law is F = ma.
#     The quadratic formula is x = (-b ± √(b^2 - 4ac)) / 2a.
#     """
#
#     result = explain_equations_in_text(sample)
#     print("\n🔎 Final Output:\n", result)

# import os
# import re
# import requests
# from gpt4all import GPT4All
# from PyPDF2 import PdfReader
#
# # ==============================
# # CONFIG
# # ==============================
# MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_0.gguf"
# MODEL_FILE = "qwen2.5-1.5b-instruct-q4_0.gguf"
# MODEL_PATH = f"models/{MODEL_FILE}"
#
# CHUNK_SIZE = 600  # words per chunk, safe for small models
#
# DEEP_PROMPT = """
# You are an expert who provides deep and accurate explanations.
#
# You MUST follow these rules exactly:
#
# 1. Only explain the meaning of the given text.
# 2. Do NOT invent, add, assume, or hallucinate any details.
# 3. Do NOT repeat sentences from the text.
# 4. Do NOT summarize.
# 5. Do NOT fix the structure.
# 6. Only explain what exists **inside the provided text**.
# 7. If the text is unclear, explain ONLY what can be understood, do not guess.
# 8. Explanation must be in a single coherent paragraph.
#
# Now provide a deep explanation of the following text:
#
# TEXT STARTS:
# {chunk}
# TEXT ENDS.
# """
#
#
# def ensure_model():
#     if not os.path.exists("models"):
#         os.makedirs("models")
#
#     if not os.path.exists(MODEL_PATH):
#         print("📥 Model not found — downloading...")
#         with requests.get(MODEL_URL, stream=True) as r:
#             with open(MODEL_PATH, "wb") as f:
#                 for chunk in r.iter_content(1024 * 1024):
#                     f.write(chunk)
#         print("✔ Download complete.")
#     else:
#         print("✔ Model already exists.")
#
#
# def load_model():
#     print("⏳ Loading model...")
#     llm = GPT4All(model_name=MODEL_FILE, model_path="models")
#     print("✔ Model loaded.")
#     return llm
#
#
# def read_pdf(file_path):
#     reader = PdfReader(file_path)
#     text_pages = []
#     for page in reader.pages:
#         text = page.extract_text() or ""
#         text = re.sub(r"[\n\r]+", " ", text)
#         text = re.sub(r"\s{2,}", " ", text)
#         text_pages.append(text.strip())
#     return text_pages
#
#
# def chunk_text(text, chunk_size=CHUNK_SIZE):
#     words = text.split()
#     for i in range(0, len(words), chunk_size):
#         yield " ".join(words[i:i + chunk_size])
#
#
# def explain_chunk(model, chunk):
#     prompt = DEEP_PROMPT.format(chunk=chunk)
#     response = model.generate(prompt, max_tokens=600)
#     return response.strip()
#
#
# def main():
#     ensure_model()
#     model = load_model()
#
#     pdf_path = input("📄 Enter PDF path: ").strip()
#     if not os.path.exists(pdf_path):
#         print("❌ File not found.")
#         return
#
#     pages = read_pdf(pdf_path)
#     print(f"✔ Loaded {len(pages)} pages.\n")
#
#     final_output = ""
#
#     for i, page in enumerate(pages, start=1):
#         print(f"🔍 Processing Page {i}/{len(pages)}...")
#         for j, chunk in enumerate(chunk_text(page), start=1):
#             print(f"   • Explaining Chunk {j}...")
#             explanation = explain_chunk(model, chunk)
#             final_output += f"\n\n{explanation}"
#
#     with open("deep_explanation.txt", "w", encoding="utf-8") as f:
#         f.write(final_output)
#
#     print("\n✅ Complete! Saved: deep_explanation.txt")
#
#
# if __name__ == "__main__":
#     main()

# import pyttsx3
#
# engine = pyttsx3.init()
# voices = engine.getProperty('voices')
#
# for v in voices:
#     print(v.id, "-", v.name)


# manual input equation

# import os
# import re
# import requests
# from gpt4all import GPT4All
#
# # =====================================
# # CONFIG
# # =====================================
#
# MODEL_NAME = "TinyLlama-1.1B-Chat-v0.3-GGUF"
# MODEL_FILE = "tinyllama.gguf"
# MODEL_PATH = f"models/{MODEL_FILE}"
#
# MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_0.gguf"
#
#
# # =====================================
# # DOWNLOAD IF NOT EXIST
# # =====================================
#
# def ensure_model_exists():
#     if not os.path.exists("models"):
#         os.makedirs("models")
#
#     if not os.path.isfile(MODEL_PATH):
#         print("📥 Model not found — downloading TinyLlama (350MB)...")
#         response = requests.get(MODEL_URL, stream=True)
#
#         with open(MODEL_PATH, "wb") as f:
#             for chunk in response.iter_content(1024 * 1024):
#                 f.write(chunk)
#
#         print("✅ TinyLlama downloaded!")
#
#     else:
#         print("✅ Model already exists — loading offline...")
#
#
# # =====================================
# # LOAD MODEL
# # =====================================
#
# ensure_model_exists()
# print("🔄 Loading TinyLlama model...")
# model = GPT4All(model_name=MODEL_FILE, model_path="models")
# print("✅ Model Loaded!")
#
#
# # =====================================
# # EQUATION EXPLAIN FUNCTION
# # =====================================
#
# def explain_equations_in_text(text: str) -> str:
#     """
#     Detect and explain equations using TinyLlama.
#     """
#
#     equation_patterns = [
#         r'([A-Za-z0-9]+ ?= ?[A-Za-z0-9\+\-\*/\^\(\)±√]+)',
#         r'(\\[a-z]+{[^}]+})',
#         r'([A-Za-z]+ ?\([A-Za-z0-9, ]*\) ?= ?[A-Za-z0-9\+\-\*/\^\(\)]+)',
#     ]
#
#     original = text
#
#     for pat in equation_patterns:
#         matches = re.findall(pat, text)
#         for eq in matches:
#
#             print(f"\n🔍 Explaining equation: {eq}\n")
#
#             prompt = f"""
# You are a math expert. Explain the following equation in simple words:
# Equation: {eq}
#
# Explain:
# 1. What does this equation mean?
# 2. Why is it used?
# 3. Where is it used in real life?
# 4. Who introduced it (if known)?
# Keep it short and easy.
# """
#
#             try:
#                 explanation = model.generate(prompt, max_tokens=200).strip()
#             except:
#                 explanation = f"[Failed to explain equation: {eq}]"
#
#             text = text.replace(eq, f"[Equation Explanation]: {explanation}")
#
#     # If no equation found, respond normal
#     if text == original:
#         return "⚠ No equation found. Please enter a valid equation."
#
#     return text
#
#
# # =====================================
# # CONTINUOUS LOOP (until user types exit)
# # =====================================
#
# print("\n==============================")
# print("🔢 Equation Explanation Assistant (Offline)")
# print("Type an equation or text containing equations.")
# print("Type 'exit' to quit.")
# print("==============================\n")
#
# while True:
#     user_input = input("\n📝 Enter equation/text: ")
#
#     if user_input.lower().strip() in ["exit", "quit"]:
#         print("👋 Exiting program. Goodbye!")
#         break
#
#     result = explain_equations_in_text(user_input)
#     print("\n🔎 Explanation:\n")
#     print(result)


# full ready hia
# import os
# import re
# import requests
# from gpt4all import GPT4All
# from PyPDF2 import PdfReader
#
# # ==============================
# # CONFIG
# # ==============================
# MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_0.gguf"
# MODEL_FILE = "qwen2.5-1.5b-instruct-q4_0.gguf"
# MODEL_PATH = f"models/{MODEL_FILE}"
#
# CHUNK_SIZE = 600  # words per chunk
#
# DEEP_PROMPT = """
# You are an expert who provides deep and accurate explanations of a single equation.
#
# Rules:
# 1. Only explain the given equation.
# 2. Do NOT invent or assume anything outside it.
# 3. Keep the explanation short: 2-3 lines.
# 4. Include creator (if known) or LLM can infer.
# 5. Explain the relationship expressed by the equation.
#
# TEXT STARTS:
# {chunk}
# TEXT ENDS.
# """
#
# # ------------------------------
# # Utilities
# # ------------------------------
# def ensure_model():
#     if not os.path.exists("models"):
#         os.makedirs("models")
#     if not os.path.exists(MODEL_PATH):
#         print("📥 Model not found — downloading...")
#         with requests.get(MODEL_URL, stream=True) as r:
#             with open(MODEL_PATH, "wb") as f:
#                 for chunk in r.iter_content(1024 * 1024):
#                     f.write(chunk)
#         print("✔ Download complete.")
#     else:
#         print("✔ Model already exists.")
#
# def load_model():
#     print("⏳ Loading model...")
#     llm = GPT4All(model_name=MODEL_FILE, model_path="models")
#     print("✔ Model loaded.")
#     return llm
#
# def sanitize_path(path):
#     """Remove hidden/unicode characters from path."""
#     return re.sub(r"[^\x20-\x7E]", "", path)
#
# def read_pdf(file_path):
#     reader = PdfReader(file_path)
#     text_pages = []
#     for page in reader.pages:
#         text = page.extract_text() or ""
#         text = re.sub(r"[\n\r]+", " ", text)
#         text = re.sub(r"\s{2,}", " ", text)
#         text_pages.append(text.strip())
#     return text_pages
#
# def read_txt(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         return [f.read()]
#
# def split_equations(text):
#     # Split equations by common delimiters (comma, semicolon, newline)
#     eqs = re.split(r",\s*|;\s*|\n", text)
#     eqs = [e.strip() for e in eqs if e.strip()]
#     return eqs
#
# def explain_equation(model, eq):
#     prompt = DEEP_PROMPT.format(chunk=eq)
#     response = model.generate(prompt, max_tokens=150)
#     return response.strip()
#
# # ------------------------------
# # Main
# # ------------------------------
# def main():
#     ensure_model()
#     model = load_model()
#
#     choice = input("Do you want to read from PDF or text file? (pdf/txt): ").strip().lower()
#     file_path = input("Enter file path: ").strip()
#     file_path = sanitize_path(file_path)
#
#     if not os.path.exists(file_path):
#         print("❌ File not found.")
#         return
#
#     if choice == "pdf":
#         pages = read_pdf(file_path)
#     else:
#         pages = read_txt(file_path)
#
#     all_equations = []
#     for page in pages:
#         all_equations.extend(split_equations(page))
#
#     print(f"✔ Found {len(all_equations)} equations.\n")
#
#     final_output = ""
#     for i, eq in enumerate(all_equations, start=1):
#         print(f"🔍 Explaining Equation {i}: {eq}")
#         explanation = explain_equation(model, eq)
#         final_output += f"\nEquation: {eq}\nExplanation: {explanation}\n"
#
#     with open("equation_explanations.txt", "w", encoding="utf-8") as f:
#         f.write(final_output)
#
#     print("\n✅ Complete! Saved: equation_explanations.txt")
#
# if __name__ == "__main__":
#     main()

# import os
import re
import requests
from gpt4all import GPT4All
from PyPDF2 import PdfReader

# ==============================
# CONFIG
# ==============================
MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_0.gguf"
MODEL_FILE = "qwen2.5-1.5b-instruct-q4_0.gguf"
MODEL_PATH = f"models/{MODEL_FILE}"

CHUNK_SIZE = 1000  # words per chunk for equations

DEEP_PROMPT = """
You are an expert who explains equations deeply.

Rules:
1. Only explain the equation provided.
2. Do NOT invent or summarize other content.
3. Give explanation in 2–3 lines.
4. If creator is not known, write 'Not specifically attributed'.

Now explain the following equation:

Equation: {equation}
"""

# ==============================
# FUNCTIONS
# ==============================
def ensure_model():
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists(MODEL_PATH):
        print("📥 Model not found — downloading...")
        with requests.get(MODEL_URL, stream=True) as r:
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
        print("✔ Download complete.")
    else:
        print("✔ Model already exists.")

def load_model():
    print("⏳ Loading model...")
    llm = GPT4All(model_name=MODEL_FILE, model_path="models")
    print("✔ Model loaded.")
    return llm

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text_pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = re.sub(r"[\n\r]+", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        text_pages.append(text.strip())
    return text_pages

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [f.read()]

def clean_path(path):
    # Remove invisible characters and extra spaces
    return path.encode("utf-8", "ignore").decode("utf-8").strip()

def extract_equations(text):
    # Match equations separated by commas or newlines, avoid splitting inside functions
    pattern = r"([A-Za-z0-9_\\^=∫\+\-\*/\(\)\[\]\,\.\s]+=[^,;]+|[A-Za-z0-9_\\^=∫\+\-\*/\(\)\[\]]+)"
    matches = re.findall(pattern, text)
    # Remove very short or irrelevant matches
    equations = [eq.strip() for eq in matches if len(eq.strip()) > 3]
    return equations

def explain_equation(model, equation):
    prompt = DEEP_PROMPT.format(equation=equation)
    response = model.generate(prompt, max_tokens=200)
    return response.strip()

# ==============================
# MAIN
# ==============================
import os

def clean_path(path):
    # Remove invisible Unicode characters
    invisible_chars = [
        '\u200b', '\u200c', '\u200d', '\u2060', '\u2061', '\u2062', '\u2063',
        '\u2064', '\u2066', '\u2067', '\u2068', '\u2069', '\u202a', '\u202b',
        '\u202c', '\u202d', '\u202e'
    ]
    for char in invisible_chars:
        path = path.replace(char, "")
    return path.strip()

def main():
    ensure_model()
    model = load_model()

    source_type = input("Do you want to read from PDF or text file? (pdf/txt): ").strip().lower()
    file_path = input("Enter file path: ").strip()
    file_path = clean_path(file_path)  # Clean the path

    # Debug: show cleaned path
    print(f"✔ Using cleaned file path: {file_path}")

    if not os.path.exists(file_path):
        print("❌ File not found.")
        return

    # Read content
    if source_type == "pdf":
        pages = read_pdf(file_path)
        content = " ".join(pages)
    elif source_type == "txt":
        pages = read_txt(file_path)
        content = " ".join(pages)
    else:
        print("❌ Invalid source type.")
        return

    # Extract equations
    equations = extract_equations(content)
    print(f"✔ Found {len(equations)} equations.\n")

    # Explain equations
    final_output = ""
    for i, eq in enumerate(equations, start=1):
        print(f"🔍 Explaining Equation {i}: {eq}")
        explanation = explain_equation(model, eq)
        final_output += f"Equation {i}: {eq}\n{explanation}\n\n"

    # Save explanations
    with open("equation_explanations.txt", "w", encoding="utf-8") as f:
        f.write(final_output)

    print("\n✅ Complete! Saved explanations to 'equation_explanations.txt'")


if __name__ == "__main__":
    main()
