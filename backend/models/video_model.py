import asyncio
import io
import tempfile
import os
import cv2
import numpy as np
from PIL import Image
from utils.gemini import verify_media_with_gemini


def extract_frames(video_bytes: bytes, num_frames: int = 8) -> list:
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            tmp_path = f.name

        cap    = cv2.VideoCapture(tmp_path)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        if total <= 0:
            cap.release()
            os.remove(tmp_path)
            return frames

        indices = np.linspace(0, total - 1, num=min(num_frames, total), dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))

        cap.release()
        os.remove(tmp_path)
        return frames

    except Exception as e:
        print(f"[VIDEO FRAME ERROR] {e}")
        return []


def analyze_frame_consistency(frames: list) -> float:
    if len(frames) < 2:
        return 0.3

    diffs = []
    for i in range(len(frames) - 1):
        arr1 = np.array(frames[i].resize((64, 64))).astype(float)
        arr2 = np.array(frames[i + 1].resize((64, 64))).astype(float)
        diff = np.mean(np.abs(arr1 - arr2)) / 255.0
        diffs.append(diff)

    mean_diff     = float(np.mean(diffs))
    std_diff      = float(np.std(diffs))
    inconsistency = min((mean_diff * 0.5) + (std_diff * 2.0), 1.0)
    return inconsistency


async def detect_video(video_bytes: bytes, filename: str) -> dict:
    loop   = asyncio.get_event_loop()
    frames = await loop.run_in_executor(None, extract_frames, video_bytes)

    if not frames:
        return {
            "modality":           "video",
            "authenticity_score": 50.0,
            "confidence_score":   20.0,
            "explanation":        "Could not extract frames from video for analysis.",
            "details": {
                "frames_analyzed":    0,
                "top_classification": "Unknown"
            }
        }

    inconsistency      = await loop.run_in_executor(None, analyze_frame_consistency, frames)
    authenticity_score = (1.0 - inconsistency) * 100
    fake_score         = inconsistency * 100

    if authenticity_score > 75:
        explanation    = (
            "Frame-by-frame analysis shows consistent temporal transitions, "
            "suggesting natural video without deepfake artifacts."
        )
        classification = "Real"
    elif authenticity_score > 45:
        explanation    = (
            "Some frame-level inconsistencies detected. This could indicate light editing "
            "or encoding artifacts. Not conclusive."
        )
        classification = "Uncertain"
    else:
        explanation    = (
            "Significant temporal inconsistencies detected across frames, "
            "consistent with deepfake video generation artifacts."
        )
        classification = "Fake"

    if frames:
        buffer = io.BytesIO()
        frames[0].save(buffer, format="JPEG")
        frame_bytes = buffer.getvalue()

        gemini = await verify_media_with_gemini("image", "frame.jpg", frame_bytes)
        if gemini and "classification" in gemini:
            if gemini["classification"] == "Fake":
                authenticity_score = min(authenticity_score, 20.0)
                classification     = "Fake"
                explanation        = gemini.get("reasoning", explanation)
            elif gemini["classification"] == "Real" and authenticity_score > 60:
                authenticity_score = max(authenticity_score, 75.0)
                classification     = "Real"

    return {
        "modality":           "video",
        "authenticity_score": round(authenticity_score, 2),
        "confidence_score":   round(max(authenticity_score, fake_score), 2),
        "explanation":        explanation,
        "details": {
            "top_classification":     classification,
            "frames_analyzed":        len(frames),
            "temporal_inconsistency": round(fake_score, 2),
        }
    }