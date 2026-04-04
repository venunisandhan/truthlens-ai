import asyncio
import io
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from utils.gemini import verify_media_with_gemini


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
    loop = asyncio.get_event_loop()

    ela_score          = await loop.run_in_executor(None, ela_analysis, image_bytes)
    authenticity_score = (1.0 - ela_score) * 100
    fake_score         = ela_score * 100
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
            "ela_manipulation_score": round(fake_score, 2),
            "ela_authentic_score":    round(authenticity_score, 2),
        }
    }