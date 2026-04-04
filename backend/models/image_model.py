import asyncio
import io
import os
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from transformers import pipeline
from utils.gemini import verify_media_with_gemini

LOCAL_IMAGE_MODEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../training/outputs/image_model"))
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "prithivMLmods/Deep-Fake-Detector-v2-Model")

class ImageDetector:
    def __init__(self):
        model_path = LOCAL_IMAGE_MODEL if os.path.exists(LOCAL_IMAGE_MODEL) else HF_IMAGE_MODEL
        print(f"[IMAGE] Loading Vision model: {model_path}")
        self.classifier = pipeline("image-classification", model=model_path)
        print("[IMAGE] Model loaded.")

    def predict(self, image: Image.Image) -> dict:
        try:
            results = self.classifier(image)
        except Exception as e:
            print(f"[IMAGE ML ERROR] {e}")
            return {"real_score": 0.5, "fake_score": 0.5}

        real_score = 0.0
        fake_score = 0.0
        for r in results:
            lbl = str(r["label"]).upper()
            if lbl in ["REAL", "HUMAN", "LABEL_0"]:
                real_score = float(r["score"])
            else:
                fake_score = float(r["score"])
        
        if real_score == 0 and fake_score == 0:
            real_score, fake_score = 0.5, 0.5
        elif real_score == 0:
            real_score = 1.0 - fake_score
        elif fake_score == 0:
            fake_score = 1.0 - real_score
            
        return {"real_score": real_score, "fake_score": fake_score}

_detector: ImageDetector = None


def ela_analysis(image_bytes: bytes) -> float:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")

        diff     = ImageChops.difference(img, compressed)
        enhancer = ImageEnhance.Brightness(diff)
        diff_enhanced = enhancer.enhance(10)

        arr       = np.array(diff_enhanced).astype(float)
        ela_score = float(np.mean(arr))
        normalized = min(ela_score / 50.0, 1.0)
        return normalized

    except Exception as e:
        print(f"[ELA ERROR] {e}")
        return 0.5


async def detect_image(image_bytes: bytes) -> dict:
    global _detector
    if _detector is None:
        _detector = ImageDetector()

    loop = asyncio.get_event_loop()

    # ELA
    ela_score          = await loop.run_in_executor(None, ela_analysis, image_bytes)
    ela_auth           = (1.0 - ela_score)
    ela_fake           = ela_score

    # ML
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    ml_result = await loop.run_in_executor(None, _detector.predict, img)
    vit_auth = ml_result["real_score"]
    vit_fake = ml_result["fake_score"]

    authenticity_score = ((vit_auth * 0.7) + (ela_auth * 0.3)) * 100
    fake_score = ((vit_fake * 0.7) + (ela_fake * 0.3)) * 100
    confidence_score   = max(authenticity_score, fake_score)

    if authenticity_score > 75:
        explanation    = (
            "Error Level Analysis shows consistent compression levels throughout the image, "
            "indicating no significant digital manipulation."
        )
        classification = "Real"
    elif authenticity_score > 45:
        explanation    = (
            "Moderate ELA inconsistencies detected. The image may have been lightly edited "
            "or re-saved multiple times. Cannot confirm deepfake with certainty."
        )
        classification = "Uncertain"
    else:
        explanation    = (
            "Significant ELA inconsistencies detected across the image. "
            "Compression artifacts suggest digital manipulation or compositing."
        )
        classification = "Fake"

    if 35.0 <= authenticity_score <= 65.0:
        gemini = await verify_media_with_gemini("image", "image.jpg", image_bytes)
        if gemini and "classification" in gemini:
            if gemini["classification"] == "Real":
                authenticity_score = max(authenticity_score, 75.0)
                classification     = "Real"
                explanation        = gemini.get("reasoning", explanation)
            elif gemini["classification"] == "Fake":
                authenticity_score = min(authenticity_score, 25.0)
                classification     = "Fake"
                explanation        = gemini.get("reasoning", explanation)

    return {
        "modality":           "image",
        "authenticity_score": round(authenticity_score, 2),
        "confidence_score":   round(confidence_score, 2),
        "explanation":        explanation,
        "details": {
            "top_classification":     classification,
            "ela_manipulation_score": round(ela_fake * 100, 2),
            "ela_authentic_score":    round(ela_auth * 100, 2),
            "vit_manipulation_score": round(vit_fake * 100, 2),
            "vit_authentic_score":    round(vit_auth * 100, 2),
        }
    }