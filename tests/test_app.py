import pytest
from PIL import Image
from src.food_recognition import recognize_and_estimate_nutrients

def test_recognize_and_estimate_nutrients():
    image = Image.new('RGB', (100, 100))
    result = recognize_and_estimate_nutrients(image)
    assert "Detected" in result
