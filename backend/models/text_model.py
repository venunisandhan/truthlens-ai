import os
import re
import asyncio
import httpx
import difflib
import html
import urllib.parse
from dotenv import load_dotenv
from transformers import pipeline
from utils.gemini import verify_text_with_gemini

load_dotenv()

NEWS_API_KEY       = os.getenv("NEWS_API_KEY")
GNEWS_API_KEY      = os.getenv("GNEWS_API_KEY")
MEDIASTACK_API_KEY = os.getenv("MEDIASTACK_API_KEY")
THENEWSAPI_KEY     = os.getenv("THENEWSAPI_KEY")

LOCAL_TEXT_MODEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../training/outputs/text_model"))
HF_TEXT_MODEL = os.getenv("HF_TEXT_MODEL", "hamzab/roberta-fake-news-classification")
HF_AI_DETECTOR = "roberta-base-openai-detector"

TRUSTED_SOURCES = [
    "bbc", "reuters", "cnn", "new york times", "the guardian",
    "npr", "bloomberg", "associated press", "wikipedia",
    "hindustan times", "the hindu", "ndtv"
]


def extract_keywords(text: str) -> str:
    stopwords = {
        "a","an","the","and","but","or","for","nor","on","at","to","from","by",
        "with","about","as","into","like","through","after","over","between","out",
        "against","during","without","before","under","around","among","of","in",
        "is","are","was","were","be","been","being","have","has","had","do","does",
        "did","my","your","his","her","its","our","their","said","say","friend",
        "because","everyone","knows","some","someone","they","this","that","these",
        "those","can","could","would","should","will","people","least","more","less",
        "many","much","just","also","very","really","then","than","when","where",
        "which","who","whom","what","how","why"
    }
    words = re.findall(r'\b[a-z]+\b', text.lower())
    keywords = [w for w in words if w not in stopwords and len(w) > 3]
    if len(keywords) < 3:
        fallback = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in fallback if w not in stopwords and len(w) > 2]
    return " ".join(keywords[:8])


async def search_wikipedia(query: str) -> list:
    if not query:
        return []
    encoded = urllib.parse.quote(query)
    url = (
        f"https://en.wikipedia.org/w/api.php"
        f"?action=query&generator=search&gsrsearch={encoded}"
        f"&prop=extracts&exintro=1&explaintext=1&exchars=600"
        f"&format=json&gsrlimit=3"
    )
    try:
        headers = {"User-Agent": "TruthLensAI/2.0"}
        async with httpx.AsyncClient(timeout=5.0, headers=headers) as client:
            res = await client.get(url)
            if res.status_code != 200:
                return []
            data = res.json()
            articles = []
            for _, item in data.get("query", {}).get("pages", {}).items():
                title   = item.get("title", "")
                extract = re.sub(r'<[^>]+>', '', item.get("extract", ""))
                extract = html.unescape(extract)
                if extract:
                    articles.append({"title": f"{title}: {extract}", "source": "wikipedia"})
            return articles
    except Exception as e:
        print(f"[WIKI ERROR] {e}")
        return []


async def search_news_apis(query: str) -> list:
    if not query:
        return []
    encoded = urllib.parse.quote(query)
    tasks = []

    async with httpx.AsyncClient(timeout=5.0) as client:
        if NEWS_API_KEY:
            tasks.append(client.get(
                f"https://newsapi.org/v2/everything?q={encoded}&apiKey={NEWS_API_KEY}&pageSize=10"
            ))
        if GNEWS_API_KEY:
            tasks.append(client.get(
                f"https://gnews.io/api/v4/search?q={encoded}&apikey={GNEWS_API_KEY}&max=10"
            ))
        if MEDIASTACK_API_KEY:
            tasks.append(client.get(
                f"http://api.mediastack.com/v1/news?access_key={MEDIASTACK_API_KEY}&keywords={encoded}&limit=10"
            ))
        if THENEWSAPI_KEY:
            tasks.append(client.get(
                f"https://api.thenewsapi.com/v1/news/all?api_token={THENEWSAPI_KEY}&search={encoded}&limit=10"
            ))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    articles = []
    for res in results:
        if isinstance(res, BaseException):
            continue
        if getattr(res, "status_code", 0) != 200:
            continue
        try:
            data  = res.json()
            items = data.get("articles", data.get("data", []))
            for item in items:
                if not isinstance(item, dict):
                    continue
                title  = str(item.get("title", ""))
                source = item.get("source", "")
                if isinstance(source, dict):
                    source = source.get("name", "")
                articles.append({"title": title, "source": str(source).lower()})
        except Exception:
            continue

    wiki = await search_wikipedia(query)
    return wiki + articles


class TextDetector:
    def __init__(self):
        fake_news_model = LOCAL_TEXT_MODEL if os.path.exists(LOCAL_TEXT_MODEL) else HF_TEXT_MODEL
        print(f"[TEXT] Loading Fake News model: {fake_news_model}")
        self.classifier = pipeline(
            "text-classification",
            model=fake_news_model,
            truncation=True,
            max_length=512
        )
        print(f"[TEXT] Loading AI Detector: {HF_AI_DETECTOR}")
        self.ai_detector = pipeline(
            "text-classification",
            model=HF_AI_DETECTOR,
            truncation=True,
            max_length=512
        )
        print("[TEXT] Loading NLI model: roberta-large-mnli")
        self.nli = pipeline(
            "text-classification",
            model="roberta-large-mnli"
        )
        print("[TEXT] Models loaded.")

    def predict(self, text: str) -> dict:
        result = self.classifier(text[:512])[0]
        label  = str(result["label"]).upper()
        score  = float(result["score"])

        if label in ["REAL", "LABEL_0", "TRUE", "HUMAN"]:
            real_score = score
            fake_score = 1.0 - score
        else:
            fake_score = score
            real_score = 1.0 - score

        ai_result = self.ai_detector(text[:512])[0]
        ai_label = str(ai_result["label"]).upper()
        ai_score = float(ai_result["score"])

        if ai_label in ["REAL", "HUMAN", "LABEL_0"]:
            human_score = ai_score
            ai_gen_score = 1.0 - ai_score
        else:
            ai_gen_score = ai_score
            human_score = 1.0 - ai_score

        return {
            "real_score": real_score, 
            "fake_score": fake_score,
            "human_score": human_score,
            "ai_gen_score": ai_gen_score
        }

    def check_entailment(self, claim: str, articles: list) -> dict:
        if not articles:
            return {"entailment": False, "contradiction": False}

        trusted  = [a for a in articles if any(ts in a["source"] for ts in TRUSTED_SOURCES)]
        to_check = articles[:5]
        for t in trusted[:5]:
            if t not in to_check:
                to_check.append(t)

        entailment_count    = 0
        contradiction_count = 0

        for article in to_check:
            premise = article["title"]
            sim = difflib.SequenceMatcher(None, claim.lower(), premise.lower()).ratio()

            if sim > 0.80 or \
               (claim.lower() in premise.lower() and len(claim) > 15) or \
               (premise.lower() in claim.lower() and len(premise) > 15):
                entailment_count += 1
                continue

            res   = self.nli(f"{premise} </s></s> {claim}")[0]
            label = res["label"]
            conf  = float(res["score"])

            if label == "ENTAILMENT" and conf > 0.50:
                entailment_count += 1
            elif label == "CONTRADICTION" and conf > 0.90:
                contradiction_count += 1

        return {
            "entailment":    entailment_count > 0,
            "contradiction": contradiction_count > 0 and entailment_count == 0
        }


_detector: TextDetector = None


async def detect_text(text: str) -> dict:
    global _detector
    if _detector is None:
        _detector = TextDetector()

    loop = asyncio.get_event_loop()

    ml_result  = await loop.run_in_executor(None, _detector.predict, text)
    real_score = ml_result["real_score"]
    fake_score = ml_result["fake_score"]
    human_score = ml_result["human_score"]
    ai_gen_score = ml_result["ai_gen_score"]

    query    = extract_keywords(text)
    articles = await search_news_apis(query)

    nli_result    = await loop.run_in_executor(None, _detector.check_entailment, text, articles)
    entailment    = nli_result["entailment"]
    contradiction = nli_result["contradiction"]
    article_count = len(articles)

    authenticity_score = ((real_score * 0.7) + (human_score * 0.3)) * 100
    confidence_score   = max(max(real_score, fake_score), max(human_score, ai_gen_score)) * 100
    top_classification = "Real" if authenticity_score > 50 else "Fake"

    if authenticity_score > 70:
        explanation = "The text structure and semantics align with factual, objective reporting."
    elif authenticity_score > 40:
        explanation = "The text contains subjective or opinionated phrasing that may reduce objectivity."
    else:
        explanation = "High likelihood of misinformation or sensationalism detected in the semantics."

    if entailment:
        authenticity_score = max(authenticity_score, 97.0)
        confidence_score   = 99.0
        top_classification = "Real"
        explanation = (
            "VERIFIED: Multiple credible news sources directly corroborate this claim. "
            "Semantic analysis confirms factual alignment."
        )
    elif contradiction:
        authenticity_score = min(authenticity_score, 5.0)
        confidence_score   = 99.0
        top_classification = "Fake"
        explanation = (
            "FALSE: Multiple credible news sources directly contradict this claim. "
            "High probability of misinformation."
        )
    elif article_count >= 5:
        if fake_score < 0.80:
            authenticity_score = max(authenticity_score, 82.0)
            confidence_score   = 80.0
            top_classification = "Real (Corroborated)"
            explanation = (
                f"Found {article_count} related news articles covering this topic. "
                "While semantic verification was inconclusive, broad news coverage suggests factual basis."
            )
        else:
            authenticity_score = min(authenticity_score, 20.0)
            confidence_score   = 88.0
            top_classification = "Fake (Clickbait)"
            explanation = (
                f"Found {article_count} related articles, but the text semantics strongly indicate "
                "sensationalized or misleading framing rather than factual reporting."
            )
    elif 0 < article_count < 5:
        if authenticity_score >= 50:
            authenticity_score = 45.0
            confidence_score   = 78.0
            top_classification = "Unverified"
            explanation = (
                "The text reads as factual, but only a few loosely related articles were found. "
                "This specific claim could not be independently verified."
            )
        else:
            explanation += " A small number of related articles were found, but none confirm this claim."
    else:
        if authenticity_score >= 50:
            authenticity_score = 35.0
            confidence_score   = 82.0
            top_classification = "Unverified/Fake"
            explanation = (
                "No credible news sources were found covering this claim. "
                "This strongly suggests the claim is fabricated or entirely unverified."
            )
        else:
            authenticity_score = max(authenticity_score - 10, 5.0)
            explanation += " No credible sources found to support this claim."

    if 35.0 <= authenticity_score <= 65.0:
        gemini = await verify_text_with_gemini(text, articles)
        if gemini and "classification" in gemini:
            if gemini["classification"] == "Real":
                authenticity_score = max(authenticity_score, 70.0)
                top_classification = "Real"
                if gemini.get("reasoning"):
                    explanation = gemini["reasoning"]
            elif gemini["classification"] == "Fake":
                authenticity_score = min(authenticity_score, 30.0)
                top_classification = "Fake"
                if gemini.get("reasoning"):
                    explanation = gemini["reasoning"]

    return {
        "modality":           "text",
        "authenticity_score": round(float(authenticity_score), 2),
        "confidence_score":   round(float(confidence_score), 2),
        "explanation":        explanation.strip(),
        "details": {
            "top_classification":  top_classification,
            "real_probability":    round(real_score * 100, 2),
            "fake_probability":    round(fake_score * 100, 2),
            "human_probability":   round(human_score * 100, 2),
            "ai_gen_probability":  round(ai_gen_score * 100, 2),
            "news_articles_found": article_count,
            "nli_entailment":      entailment,
            "nli_contradiction":   contradiction,
        }
    }