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

app = FastAPI()

# Load FaceNet TFLite model
print("[INFO] Loading FaceNet model...")
interpreter = Interpreter(model_path="models/facenet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("[INFO] Model loaded successfully.")

INPUT_SIZE = 160
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

def verify_ic_structure_orb(image_np: np.ndarray, threshold: float = 15.0, use_template=False) -> bool:
    if not use_template:
        print("[INFO] Skipping template matching — generic ID mode.")
        return True

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
