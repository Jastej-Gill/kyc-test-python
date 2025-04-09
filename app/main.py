from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from PIL import Image
from tensorflow.lite.python.interpreter import Interpreter

app = FastAPI()

# Load FaceNet TFLite model
interpreter = Interpreter(model_path="models/facenet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_SIZE = 160  # FaceNet input size

def preprocess_image(image_bytes) -> np.ndarray:
    image = Image.open(image_bytes).convert('RGB')
    image_np = np.array(image)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        raise Exception("No face detected in the image")

    # Choose the largest face detected
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face_img = image_np[y:y+h, x:x+w]

    # Resize and normalize
    face_resized = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE))
    normalized = (face_resized.astype(np.float32) - 127.5) / 128.0
    return np.expand_dims(normalized, axis=0)

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
        ic_face = preprocess_image(ic_image.file)
        selfie_face = preprocess_image(selfie_image.file)

        emb1 = get_embedding(ic_face)
        emb2 = get_embedding(selfie_face)

        similarity = cosine_similarity(emb1, emb2)
        return {
            "similarity": similarity,
            "match": similarity > 0.5  # Adjust threshold as needed
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
