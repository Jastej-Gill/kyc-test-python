
---

## 📁 `README_BACKEND.md`

```markdown
# FastAPI eKYC Backend

This backend powers the Flutter eKYC app, providing:

- 🧠 Face verification (IC vs selfie) using TFlite model
- 👀 Liveness detection via multi-frame face movement and blinking
- 📄 IC OCR using EasyOCR

## 🧪 Requirements

- Python 3.10
- pip or poetry
- virtualenv (recommended)

## 🔧 Setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000
```
