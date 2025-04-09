from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
import uuid
from PIL import Image
from tensorflow.lite.python.interpreter import Interpreter

app = FastAPI()

# Load FaceNet TFLite model
interpreter = Interpreter(model_path="models/facenet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_SIZE = 160  # FaceNet expects 160x160 input

def preprocess_image(image_bytes) -> np.ndarray:
    image = Image.open(image_bytes).convert('RGB')
    image = image.resize((INPUT_SIZE, INPUT_SIZE))
    image_np = np.asarray(image).astype(np.float32)
    normalized = (image_np - 127.5) / 128.0
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
