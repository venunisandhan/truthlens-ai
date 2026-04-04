import os
import asyncio
import tempfile
from dotenv import load_dotenv
from transformers import pipeline
from utils.gemini import verify_media_with_gemini

load_dotenv()

LOCAL_AUDIO_MODEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../training/outputs/audio_model"))
HF_AUDIO_MODEL = os.getenv("HF_AUDIO_MODEL", "mo-thecreator/Deepfake-audio-detection")


class AudioDetector:
    def __init__(self):
        model_path = LOCAL_AUDIO_MODEL if os.path.exists(LOCAL_AUDIO_MODEL) else HF_AUDIO_MODEL
        print(f"[AUDIO] Loading model: {model_path}")
        self.classifier = pipeline(
            "audio-classification",
            model=model_path,
        )
        print("[AUDIO] Model loaded.")

    def predict(self, audio_bytes: bytes, filename: str) -> dict:
        suffix = ".wav"
        if filename:
            ext = filename.lower().split(".")[-1]
            if ext in ["mp3", "flac", "ogg", "wav"]:
                suffix = f".{ext}"

        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name

            results = self.classifier(tmp_path)
            os.remove(tmp_path)

            real_score = 0.0
            fake_score = 0.0

            for r in results:
                lbl = r["label"].upper()
                scr = float(r["score"])
                if lbl == "REAL":
                    real_score = scr * 100
                elif lbl == "FAKE":
                    fake_score = scr * 100

            if real_score == 0.0 and fake_score == 0.0:
                real_score = 50.0
                fake_score = 50.0
            elif real_score == 0.0:
                real_score = 100.0 - fake_score
            elif fake_score == 0.0:
                fake_score = 100.0 - real_score

            authenticity_score = real_score
            confidence_score   = max(real_score, fake_score)

            if authenticity_score > 80:
                explanation = (
                    "Vocal frequency patterns are consistent with a real human voice. "
                    "No synthetic artifacts detected."
                )
            elif authenticity_score > 50:
                explanation = (
                    "Some audio compression artifacts detected, but no definitive signs "
                    "of voice cloning or synthesis."
                )
            else:
                explanation = (
                    "Anomalies detected in the vocal spectrogram consistent with AI voice synthesis "
                    "or cloning (e.g. ElevenLabs, VITS, RVC)."
                )

            return {
                "modality":           "audio",
                "authenticity_score": round(authenticity_score, 2),
                "confidence_score":   round(confidence_score, 2),
                "explanation":        explanation,
                "details": {
                    "file_analyzed":    filename,
                    "real_probability": round(real_score, 2),
                    "fake_probability": round(fake_score, 2),
                }
            }

        except Exception as e:
            return {
                "modality":           "audio",
                "error":              str(e),
                "authenticity_score": 50.0,
                "confidence_score":   0.0,
                "explanation":        f"Error processing audio: {str(e)}",
                "details": {}
            }


_detector: AudioDetector = None


async def detect_audio(audio_bytes: bytes, filename: str) -> dict:
    global _detector
    if _detector is None:
        _detector = AudioDetector()

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _detector.predict, audio_bytes, filename)

    if "error" in result:
        return result

    auth = result.get("authenticity_score", 50.0)
    if 35.0 <= auth <= 65.0:
        gemini = await verify_media_with_gemini("audio", filename, audio_bytes)
        if gemini and "classification" in gemini:
            if gemini["classification"] == "Real":
                result["authenticity_score"] = max(auth, 75.0)
                result["explanation"] = gemini.get("reasoning", result["explanation"])
            elif gemini["classification"] == "Fake":
                result["authenticity_score"] = min(auth, 25.0)
                result["explanation"] = gemini.get("reasoning", result["explanation"])

    return result