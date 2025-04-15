from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image, ImageOps, ExifTags
from io import BytesIO
from tensorflow.lite.python.interpreter import Interpreter
import os
import uuid
import easyocr
import re
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

app = FastAPI()

# Load FaceNet TFLite model
print("[INFO] Loading FaceNet model...")
interpreter = Interpreter(model_path="models/facenet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("[INFO] Model loaded successfully.")

INPUT_SIZE = 160

print("[INFO] Initializing EasyOCR reader...")
reader = easyocr.Reader(['en', 'ms'])
print("[INFO] EasyOCR ready.")

def correct_image_rotation(image: Image.Image) -> Image.Image:
    try:
        return ImageOps.exif_transpose(image)
    except Exception as e:
        print(f"[WARN] Rotation correction skipped: {e}")
        return image

def extract_text_from_ic(image_np: np.ndarray) -> List[str]:
    print("[INFO] Running OCR on IC image...")
    results = reader.readtext(image_np)
    lines = [res[1].strip() for res in results if res[1].strip()]
    print(f"[DEBUG] OCR lines: {lines}")
    return lines

def preprocess_image(image_bytes, label: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    print(f"[INFO] Preprocessing image for label: {label}")
    raw_image = Image.open(image_bytes).convert('RGB')
    image = correct_image_rotation(raw_image)
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("[WARN] No face found in image.")
        return None, None

    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face_img = image_np[y:y+h, x:x+w]

    face_resized = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE))
    normalized = (face_resized.astype(np.float32) - 127.5) / 128.0
    return np.expand_dims(normalized, axis=0), None

def get_embedding(image_tensor: np.ndarray) -> np.ndarray:
    interpreter.set_tensor(input_details[0]['index'], image_tensor)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_details[0]['index'])
    return embedding[0]

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return float(dot / (norm_a * norm_b))

def find_best_template_match(image_np: np.ndarray, threshold: int = 30) -> dict:
    TEMPLATES = {
        "malaysia_ic": "templates/malaysia_ic_template.png",
        "malaysia_license": "templates/malaysia_licence_template.png",
        "passport": "templates/passport_template.jpg",
        "universal_id": "templates/universal_id_template.jpg",
        "universal_license": "templates/universal_licence_template.jpg"
    }

    results = []

    for doc_type, path in TEMPLATES.items():
        if not os.path.exists(path):
            continue

        template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        gray_input = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        resized_input = cv2.resize(gray_input, (600, 400))
        resized_template = cv2.resize(template, (600, 400))

        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(resized_template, None)
        kp2, des2 = orb.detectAndCompute(resized_input, None)

        orb_score = 0
        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            good_matches = [m for m in matches if m.distance < 60]
            orb_score = len(good_matches)

        ssim_score = ssim(resized_template, resized_input)

        results.append({
            "type": doc_type,
            "orb_score": orb_score,
            "ssim_score": ssim_score
        })

    best = sorted(results, key=lambda r: (r["orb_score"], r["ssim_score"]), reverse=True)[0]
    return best

@app.post("/verify_ic_structure/")
async def verify_ic_structure(ic_image: UploadFile = File(...)):
    try:
        print("[INFO] Verifying ID structure and face presence...")
        raw_ic = Image.open(ic_image.file).convert("RGB")
        image = correct_image_rotation(raw_ic)
        image_np = np.array(image)

        def detect_face(image_np: np.ndarray) -> bool:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            print(f"[INFO] Detected {len(faces)} face(s)")
            return len(faces) > 0

        if not detect_face(image_np):
            raise Exception("No face detected in the uploaded ID image.")

        match_result = find_best_template_match(image_np)
        print(f"[INFO] Best match: {match_result['type']} (ORB: {match_result['orb_score']}, SSIM: {match_result['ssim_score']:.2f})")

        if match_result["orb_score"] < 35 or match_result["ssim_score"] < 0.25:
            raise Exception("Uploaded image does not match any known ID layout.")

        lines = extract_text_from_ic(image_np)
        if len(lines) < 3:
            raise Exception("Image does not contain enough readable text to be considered an ID.")

        keywords = ["passport", "license", "id", "permit", "kad", "nombor", "nama", "name"]
        keyword_hits = any(any(word in line.lower() for word in keywords) for line in lines)

        if not keyword_hits:
            print("[WARN] No strong ID-related keywords found, but continuing since text exists.")

        return {
            "success": True,
            "message": "ID structure and facial presence validated.",
            "data": {
                "face_detected": True,
                "text_lines_detected": len(lines),
                "sample_text": lines[:5],
                "detected_document_type": match_result["type"],
                "orb_match_score": match_result["orb_score"],
                "ssim_score": round(match_result["ssim_score"], 3)
            }
        }

    except Exception as e:
        print(f"[ERROR] ID structure verification failed: {e}")
        return JSONResponse(status_code=400, content={
            "success": False,
            "message": str(e)
        })


def auto_rotate_image(pil_img: Image.Image) -> Image.Image:
    try:
        exif = pil_img._getexif()
        if exif:
            orientation_key = next(
                k for k, v in ExifTags.TAGS.items() if v == 'Orientation'
            )
            orientation = exif.get(orientation_key)

            if orientation == 3:
                pil_img = pil_img.rotate(180, expand=True)
            elif orientation == 6:
                pil_img = pil_img.rotate(270, expand=True)
            elif orientation == 8:
                pil_img = pil_img.rotate(90, expand=True)
    except Exception as e:
        print(f"[WARN] EXIF rotation skipped: {e}")
    return pil_img

@app.post("/verify_liveness_and_match/")
async def verify_liveness_and_match(
    ic_image: UploadFile = File(...),
    selfie_images: List[UploadFile] = File(...)
):
    try:
        print("[INFO] Starting liveness + face match check...")

        if len(selfie_images) != 5:
            raise Exception("Exactly 5 selfie frames are required.")

        ic_face, _ = preprocess_image(ic_image.file, label="ic")
        if ic_face is None:
            raise Exception("No face detected in IC image.")
        ic_embedding = get_embedding(ic_face)

        selfie_embeddings = []
        face_boxes = []
        eye_ratios = []

        for i, selfie in enumerate(selfie_images):
            try:
                selfie_bytes = await selfie.read()
                selfie.file.seek(0)

                print(f"[DEBUG] Selfie frame {i+1}: {len(selfie_bytes)} bytes")
                if len(selfie_bytes) < 1000:
                    print(f"[WARN] Selfie frame {i+1} might be empty or corrupted.")

                image = Image.open(BytesIO(selfie_bytes))
                image = auto_rotate_image(image)
                image = image.convert("RGB")

                image_np = np.array(image)
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)

                if len(faces) == 0:
                    print("[WARN] No face detected in one of the selfie frames.")
                    continue

                x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                face_boxes.append((x, y))
                face_crop = image_np[y:y+h, x:x+w]

                face_resized = cv2.resize(face_crop, (160, 160))
                normalized = (face_resized.astype(np.float32) - 127.5) / 128.0
                input_tensor = np.expand_dims(normalized, axis=0)
                emb = get_embedding(input_tensor)
                selfie_embeddings.append(emb)

                eye_ratios.append(w / h)

            except Exception as e:
                print(f"[ERROR] Frame {i+1} failed: {e}")
                continue

        if len(selfie_embeddings) < 2:
            raise Exception("At least 2 valid selfie frames required.")

        movement = any(
            abs(face_boxes[i][0] - face_boxes[i-1][0]) > 5 or
            abs(face_boxes[i][1] - face_boxes[i-1][1]) > 5
            for i in range(1, len(face_boxes))
        )
        eye_changes = any(
            abs(eye_ratios[i] - eye_ratios[i - 1]) > 0.05
            for i in range(1, len(eye_ratios))
        )
        liveness = movement or eye_changes

        avg_selfie_emb = np.mean(np.array(selfie_embeddings), axis=0)
        similarity = cosine_similarity(ic_embedding, avg_selfie_emb)
        match = similarity > 0.5

        print(f"[INFO] Match: {match}, Liveness: {liveness}, Similarity: {similarity}")

        return {
            "success": True,
            "message": "Liveness and face match verification completed.",
            "data": {
                "liveness_passed": liveness,
                "match": match,
                "similarity": similarity,
                "frames_processed": len(selfie_embeddings)
            }
        }

    except Exception as e:
        print(f"[ERROR] Combined verification failed: {e}")
        return JSONResponse(status_code=500, content={
            "success": False,
            "message": "Liveness and match verification failed.",
            "error": str(e)
        })
