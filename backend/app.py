from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from models.text_model import detect_text
from models.audio_model import detect_audio
from models.image_model import detect_image
from models.video_model import detect_video

app = FastAPI(
    title="TruthLens AI",
    description="Multi-Modal Misinformation and Deepfake Detection API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-project.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ok", "message": "TruthLens AI is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/api/v1/detect/text")
async def analyze_text(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(request.text) > 10000:
        raise HTTPException(status_code=400, detail="Text too long, max 10000 characters")
    return await detect_text(request.text)

@app.post("/api/v1/detect/audio")
async def analyze_audio(file: UploadFile = File(...)):
    allowed = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg",
               "audio/flac", "audio/x-wav", "audio/x-flac"]
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported audio type: {file.content_type}")
    file_bytes = await file.read()
    if len(file_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Audio file too large, max 50MB")
    return await detect_audio(file_bytes, file.filename)

@app.post("/api/v1/detect/image")
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    file_bytes = await file.read()
    if len(file_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large, max 20MB")
    return await detect_image(file_bytes)

@app.post("/api/v1/detect/video")
async def analyze_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    file_bytes = await file.read()
    if len(file_bytes) > 200 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Video too large, max 200MB")
    return await detect_video(file_bytes, file.filename)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)