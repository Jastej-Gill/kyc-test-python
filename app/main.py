from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import numpy as np
import cv2
from PIL import Image
from tensorflow.lite.python.interpreter import Interpreter
import os
import uuid
import easyocr
import re
import tempfile
import shutil

app = FastAPI()

# Load FaceNet TFLite model
interpreter = Interpreter(model_path="models/facenet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_SIZE = 160  # FaceNet input size
SAVE_DIR = "detected_faces"
os.makedirs(SAVE_DIR, exist_ok=True)

reader = easyocr.Reader(['en', 'ms'])  # EasyOCR for Malay IC

def preprocess_image(image_bytes, label: str) -> tuple[np.ndarray, str]:
    image = Image.open(image_bytes).convert('RGB')
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        raise Exception("No face detected in the image")

    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face_img = image_np[y:y+h, x:x+w]

    # Save image with bounding box
    cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
    filename = f"detected_{label}_{uuid.uuid4().hex[:8]}.jpg"
    save_path = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

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

@app.post("/verify/")
async def verify(ic_image: UploadFile = File(...), selfie_image: UploadFile = File(...)):
    try:
        ic_face, ic_path = preprocess_image(ic_image.file, "ic")
        selfie_face, selfie_path = preprocess_image(selfie_image.file, "selfie")

        emb1 = get_embedding(ic_face)
        emb2 = get_embedding(selfie_face)

        similarity = cosine_similarity(emb1, emb2)

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
        return JSONResponse(status_code=500, content={
            "success": False,
            "message": "Verification failed.",
            "error": str(e)
        })

@app.post("/extract_ic_text/")
async def extract_ic_text(ic_image: UploadFile = File(...)):
    try:
        image = Image.open(ic_image.file).convert("RGB")
        image_np = np.array(image)

        results = reader.readtext(image_np)
        lines = [res[1].strip() for res in results if res[1].strip()]
        full_text = "\n".join(lines)

        ic_number = None
        ic_index = -1
        for i, line in enumerate(lines):
            match = re.search(r'\d{6}-\d{2}-\d{4}', line)
            if match:
                ic_number = match.group(0)
                ic_index = i
                break

        name_lines = lines[ic_index + 1:ic_index + 3] if ic_index >= 0 else []
        name = " ".join(name_lines).strip().title()

        address_start = ic_index + 3
        address_lines = lines[address_start:address_start + 3]
        address = "\n".join(address_lines).strip()

        return {
            "success": True,
            "message": "IC text extracted successfully.",
            "data": {
                "ic_number": ic_number,
                "name": name,
                "address": address,
                "raw_text": full_text
            }
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "message": "Failed to extract IC text.",
            "error": str(e)
        })

@app.post("/verify_liveness/")
async def verify_liveness(
    ic_image: UploadFile = File(...),
    selfie_images: List[UploadFile] = File(...)
):
    try:
        ic_face, _ = preprocess_image(ic_image.file, label="ic")
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
                    continue

                x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                face_boxes.append((x, y))

                face_crop = image_np[y:y+h, x:x+w]
                face_resized = cv2.resize(face_crop, (160, 160))
                normalized = (face_resized.astype(np.float32) - 127.5) / 128.0
                input_tensor = np.expand_dims(normalized, axis=0)

                emb = get_embedding(input_tensor)
                selfie_embeddings.append(emb)

            except Exception:
                continue

        if len(selfie_embeddings) < 2:
            return {
                "success": False,
                "message": "Not enough valid selfie frames with detected faces.",
                "error": "At least 2 frames with face detection required."
            }

        movement = any(
            abs(face_boxes[i][0] - face_boxes[i-1][0]) > 5 or
            abs(face_boxes[i][1] - face_boxes[i-1][1]) > 5
            for i in range(1, len(face_boxes))
        )

        avg_selfie_emb = np.mean(np.array(selfie_embeddings), axis=0)
        similarity = cosine_similarity(ic_embedding, avg_selfie_emb)
        match = similarity > 0.5

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
        return JSONResponse(status_code=500, content={
            "success": False,
            "message": "Liveness verification failed.",
            "error": str(e)
        })
