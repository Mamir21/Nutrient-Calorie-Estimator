from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import requests

model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
API_KEY = "YOUR_USDA_API_KEY"

def fetch_nutritional_info(food_item):
    url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={food_item}&api_key={API_KEY}"
    response = requests.get(url)
    data = response.json()
    if 'foods' in data and len(data['foods']) > 0:
        nutrients = data['foods'][0]['foodNutrients']
        return {nutrient['nutrientName']: nutrient['value'] for nutrient in nutrients}
    return {}

def estimate_nutrients(food_item, portion_size):
    nutrients = fetch_nutritional_info(food_item)
    for nutrient in nutrients:
        nutrients[nutrient] = nutrients[nutrient] * portion_size / 100
    return nutrients

def estimate_food_weight(image):
    # Placeholder for actual weight estimation logic using a pre-trained model
    # For example, you can use a regression model trained to predict weight
    estimated_weight = 150  # Assume the model estimates 150 grams
    return estimated_weight

def recognize_and_estimate_nutrients(image):
    # The Image class from PIL is used to handle the image file
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    detected_food = None
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score > 0.9:
            detected_food = model.config.id2label[label.item()]
            break

    if detected_food:
        portion_size = estimate_food_weight(image)  # Estimate the weight
        nutrients = estimate_nutrients(detected_food, portion_size)
        nutrient_info = ", ".join([f"{k}: {v}g" for k, v in nutrients.items()])
        return f"Detected: {detected_food}, Estimated Weight: {portion_size}g, Estimated Nutrients: {nutrient_info}"
    else:
        return "No food item detected with sufficient confidence."
