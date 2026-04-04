import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


async def verify_text_with_gemini(text: str, context: list = None) -> dict:
    if not GEMINI_API_KEY:
        return {}

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        context_str = ""
        if context:
            headlines = []
            for c in context[:10]:
                title  = c.get("title", "")
                source = c.get("source", "unknown")
                if title:
                    headlines.append(f"- [{source}] {title}")
            if headlines:
                context_str = (
                    "\nRecent news headlines (use as ground truth):\n"
                    + "\n".join(headlines)
                    + "\n"
                )

        prompt = f"""You are an expert fact-checker.
Analyze the claim below and return ONLY valid JSON — no markdown, no extra text.
{context_str}
Format:
{{"classification": "Real" or "Fake", "confidence": <float 0-100>, "reasoning": "<one sentence>"}}

Claim: "{text}"
"""
        response = await model.generate_content_async(prompt)
        raw = response.text.strip().lstrip("```json").rstrip("```").strip()
        return json.loads(raw)

    except Exception as e:
        print(f"[GEMINI TEXT ERROR] {e}")
        return {}


async def verify_media_with_gemini(modality: str, filename: str, media_bytes: bytes = None) -> dict:
    if not GEMINI_API_KEY or not media_bytes:
        return {}

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        mime_map = {
            "audio": "audio/wav",
            "image": "image/jpeg",
            "video": "video/mp4",
        }

        if filename:
            ext = filename.lower().split(".")[-1]
            ext_mime = {
                "mp3": "audio/mp3", "wav": "audio/wav", "flac": "audio/flac",
                "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "mp4": "video/mp4", "webm": "video/webm",
            }
            mime_type = ext_mime.get(ext, mime_map.get(modality, "application/octet-stream"))
        else:
            mime_type = mime_map.get(modality, "application/octet-stream")

        prompt = (
            f"You are a deepfake detection expert. Analyze this {modality}. "
            "Determine if it is 'Real' or 'Fake'. "
            "Return ONLY valid JSON — no markdown, no extra text.\n"
            'Format: {"classification": "Real" or "Fake", "reasoning": "<one sentence>"}'
        )

        response = await model.generate_content_async([
            prompt,
            {"mime_type": mime_type, "data": media_bytes}
        ])
        raw = response.text.strip().lstrip("```json").rstrip("```").strip()
        return json.loads(raw)

    except Exception as e:
        print(f"[GEMINI {modality.upper()} ERROR] {e}")
        return {}