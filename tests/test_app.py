import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from PIL import Image
from food_recognition import recognize_and_estimate_nutrients, recognize_and_estimate_with_focus

def test_recognize_and_estimate_nutrients_no_food():
    image = Image.new('RGB', (100, 100))
    result = recognize_and_estimate_nutrients(image)
    assert "No food items detected with sufficient confidence." in result

def test_recognize_and_estimate_nutrients_with_food():
    image = Image.new('RGB', (800, 600), color=(73, 109, 137))
    result = recognize_and_estimate_nutrients(image)
    assert "Detected" in result or "No food items detected with sufficient confidence." in result

def test_recognize_and_estimate_with_focus():
    image = Image.new('RGB', (800, 600), color=(73, 109, 137))
    central_fraction = 0.7
    nutrient_estimation, image_with_focus = recognize_and_estimate_with_focus(image, central_fraction)
    assert isinstance(nutrient_estimation, str)
    assert isinstance(image_with_focus, Image.Image)

def test_fetch_nutritional_info():
    from food_recognition import fetch_nutritional_info
    food_item = "apple"
    result = fetch_nutritional_info(food_item)
    assert isinstance(result, dict)

def test_estimate_nutrients():
    from food_recognition import estimate_nutrients
    food_item = "apple"
    portion_size = 100
    result = estimate_nutrients(food_item, portion_size)
    assert isinstance(result, dict)
    assert "Energy" in result

def test_estimate_food_weight():
    from food_recognition import estimate_food_weight
    box = [0, 0, 50, 50]
    image_size = [100, 100]
    weight = estimate_food_weight(box, image_size)
    assert isinstance(weight, float)
