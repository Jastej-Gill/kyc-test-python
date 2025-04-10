from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import numpy as np
import cv2
from PIL import Image, ImageOps
from tensorflow.lite.python.interpreter import Interpreter
import os
import uuid
import easyocr
import re

app = FastAPI()

# Load FaceNet TFLite model
print("[INFO] Loading FaceNet model...")
interpreter = Interpreter(model_path="models/facenet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("[INFO] Model loaded successfully.")

INPUT_SIZE = 160  # FaceNet input size
SAVE_DIR = "detected_faces"
os.makedirs(SAVE_DIR, exist_ok=True)

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

def preprocess_image(image_bytes, label: str) -> tuple[np.ndarray | None, str | None]:
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

    filename = f"detected_{label}_{uuid.uuid4().hex[:8]}.jpg"
    save_path = os.path.join(SAVE_DIR, filename)
    cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Saved annotated image to {save_path}")

    face_resized = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE))
    normalized = (face_resized.astype(np.float32) - 127.5) / 128.0
    return np.expand_dims(normalized, axis=0), save_path

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

def verify_ic_structure_orb(image_np: np.ndarray, threshold: float = 15.0) -> bool:
    template_path = "templates/ic_template.jpg"
    if not os.path.exists(template_path):
        print("[WARN] IC template not found — skipping ORB matching")
        return True

    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise Exception("Failed to load IC template image.")

    gray_input = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(gray_input, None)

    if des1 is None or des2 is None:
        print("[WARN] No descriptors found in template or input image.")
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good_matches = [m for m in matches if m.distance < 60]
    match_score = len(good_matches)

    print(f"[INFO] ORB match count: {match_score}")

    vis_path = "debug_orb_matches.jpg"
    match_vis = cv2.drawMatches(template, kp1, gray_input, kp2, good_matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(vis_path, match_vis)
    print(f"[INFO] Saved ORB match visualization to {vis_path}")

    return match_score >= threshold

@app.post("/verify_ic_structure/")
async def verify_ic_structure(ic_image: UploadFile = File(...)):
    try:
        print("[INFO] Verifying IC structure only...")
        raw_ic = Image.open(ic_image.file).convert("RGB")
        image_np = np.array(correct_image_rotation(raw_ic))

        if not verify_ic_structure_orb(image_np):
            raise Exception("Uploaded image does not match the structure of a Malaysian MyKad.")

        # OCR-based discard check
        lines = extract_text_from_ic(image_np)
        has_text = any(re.search(r'[A-Z]{2,}|\d{6,}', line) for line in lines)
        if not has_text:
            raise Exception("IC image does not appear to contain valid text content.")

        return {"success": True, "message": "IC structure and text presence validated."}

    except Exception as e:
        print(f"[ERROR] IC structure verification failed: {e}")
        return JSONResponse(status_code=400, content={
            "success": False,
            "message": str(e)
        })

@app.post("/verify_face_match/")
async def verify_face_match(ic_image: UploadFile = File(...), selfie_image: UploadFile = File(...)):
    try:
        print("[INFO] Verifying face match between IC and selfie...")
        ic_face, ic_path = preprocess_image(ic_image.file, "ic")
        selfie_face, selfie_path = preprocess_image(selfie_image.file, "selfie")

        if ic_face is None or selfie_face is None:
            raise Exception("Face not detected in one of the images.")

        emb1 = get_embedding(ic_face)
        emb2 = get_embedding(selfie_face)

        similarity = cosine_similarity(emb1, emb2)
        print(f"[INFO] Similarity: {similarity}")

        return {
            "success": True,
            "message": "Face verification complete.",
            "data": {
                "similarity": similarity,
                "match": similarity > 0.5,
                "saved_files": {
                    "ic_image_with_box": ic_path,
                    "selfie_image_with_box": selfie_path
                }
            }
        }

    except Exception as e:
        print(f"[ERROR] Face match verification failed: {e}")
        return JSONResponse(status_code=500, content={
            "success": False,
            "message": "Face match verification failed.",
            "error": str(e)
        })

@app.post("/verify_liveness/")
async def verify_liveness(ic_image: UploadFile = File(...), selfie_images: List[UploadFile] = File(...)):
    try:
        print("[INFO] Starting liveness check...")
        ic_face, _ = preprocess_image(ic_image.file, label="ic")
        if ic_face is None:
            raise Exception("No face detected in IC image.")
        ic_embedding = get_embedding(ic_face)

        selfie_embeddings = []
        face_boxes = []

        for selfie in selfie_images:
            try:
                image = Image.open(selfie.file).convert("RGB")
                image_np = np.array(image)
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
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

            except Exception as frame_err:
                print(f"[ERROR] Frame processing failed: {frame_err}")
                continue

        if len(selfie_embeddings) < 2:
            raise Exception("At least 2 valid selfie frames required.")

        movement = any(
            abs(face_boxes[i][0] - face_boxes[i-1][0]) > 5 or
            abs(face_boxes[i][1] - face_boxes[i-1][1]) > 5
            for i in range(1, len(face_boxes))
        )

        avg_selfie_emb = np.mean(np.array(selfie_embeddings), axis=0)
        similarity = cosine_similarity(ic_embedding, avg_selfie_emb)
        match = similarity > 0.5

        print(f"[INFO] Liveness passed: {movement}, Similarity: {similarity}, Match: {match}")

        return {
            "success": True,
            "message": "Liveness and match verification completed.",
            "data": {
                "similarity": similarity,
                "match": match,
                "liveness_passed": movement,
                "frames_processed": len(selfie_embeddings)
            }
        }

    except Exception as e:
        print(f"[ERROR] Liveness verification failed: {e}")
        return JSONResponse(status_code=500, content={
            "success": False,
            "message": "Liveness verification failed.",
            "error": str(e)
        })
