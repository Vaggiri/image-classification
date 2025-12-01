# llm_server.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from moondream import Moondream
import uvicorn
from PIL import Image
import io
import os

app = FastAPI()

# config
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 5_000_000))  # 5 MB default

# load model (this may download on first run if not preloaded)
MODEL_CACHE_DIR = os.environ.get("MOONDREAM_CACHE_DIR", None)
if MODEL_CACHE_DIR:
    # if your Moondream package honors cache_dir, pass it; otherwise this is a hint.
    os.environ["MOONDREAM_CACHE_DIR"] = MODEL_CACHE_DIR

print("Loading Moondream model (this may take a while)...")
model = Moondream.from_pretrained("vikhyatk/moondream2")
model.eval()
print("Model loaded")

@app.post("/detect")
async def detect_plastic(file: UploadFile = File(...)):
    img_bytes = await file.read()
    if len(img_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    prompt = """
    You are a vision classifier. Look at the image and tell ONLY:
    1 if the object OR debris OR floating item looks like PLASTIC,
    0 if it is NOT plastic (leaf, wood, foam, natural object).
    Respond with only a single digit: 1 or 0.
    """

    answer = model.query(prompt, image=img).strip()
    if "1" in answer:
        return {"plastic": 1}
    else:
        return {"plastic": 0}

if __name__ == "__main__":
    uvicorn.run("llm_server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
