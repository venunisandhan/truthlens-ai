import asyncio
from models.text_model import TextDetector
from models.image_model import ImageDetector
from models.audio_model import AudioDetector

def test_models():
    print("Testing TextDetector initialization...")
    text_det = TextDetector()
    print("TextDetector OK.\n")

    print("Testing ImageDetector initialization...")
    img_det = ImageDetector()
    print("ImageDetector OK.\n")

    print("Testing AudioDetector initialization...")
    aud_det = AudioDetector()
    print("AudioDetector OK.\n")

    print("All models successfully initialized!")

if __name__ == "__main__":
    test_models()
