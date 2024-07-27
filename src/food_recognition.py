from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw
import torch
import requests
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("USDA_API_KEY")
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')

def fetch_nutritional_info(food_item):
    url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={food_item}&api_key={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'foods' in data and len(data['foods']) > 0:
            nutrients = data['foods'][0]['foodNutrients']
            return {nutrient['nutrientName']: nutrient['value'] for nutrient in nutrients}
        return {}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching nutritional info for {food_item}: {e}")
        return {}

def estimate_nutrients(food_item, portion_size):
    nutrients = fetch_nutritional_info(food_item)
    essential_nutrients = {
        "Energy": 0,
        "Protein": 0,
        "Carbohydrate, by difference": 0,
        "Total lipid (fat)": 0
    }
    for nutrient, value in nutrients.items():
        if nutrient in essential_nutrients:
            essential_nutrients[nutrient] = value * portion_size / 100
    return essential_nutrients

def estimate_food_weight(box, image_size):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    image_area = image_size[0] * image_size[1]
    estimated_weight = 300 * (box_area / image_area)  
    return estimated_weight

def preprocess_image(image):
    return image.resize((800, 600))

def get_focus_area(image_size, central_fraction=0.7):
    x_center, y_center = image_size[0] / 2, image_size[1] / 2
    central_width, central_height = image_size[0] * central_fraction, image_size[1] * central_fraction
    focus_area = (
        x_center - central_width / 2,
        y_center - central_height / 2,
        x_center + central_width / 2,
        y_center + central_height / 2
    )
    return focus_area

def draw_focus_area(image, focus_area):
    draw = ImageDraw.Draw(image)
    draw.rectangle(focus_area, outline="red", width=3)
    return image

def is_food_item(label):
    # Add more food categories as needed
    food_categories = ['food', 'fruit', 'vegetable', 'meat', 'dish', 'dessert', 'bread', 'sandwich']
    return any(category in label.lower() for category in food_categories)

def recognize_and_estimate_nutrients(image, central_fraction=0.7):
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = preprocess_image(image)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    detected_foods = []
    total_nutrients = {
        "Energy": 0,
        "Protein": 0,
        "Carbohydrate, by difference": 0,
        "Total lipid (fat)": 0
    }

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        food_label = model.config.id2label[label.item()]
        if score > 0.5 and is_food_item(food_label):  
            portion_size = estimate_food_weight(box, image.size)
            nutrients = estimate_nutrients(food_label, portion_size)
            detected_foods.append({
                "label": food_label,
                "weight": portion_size,
                "nutrients": nutrients
            })
            for nutrient, value in nutrients.items():
                total_nutrients[nutrient] += value

    if detected_foods:
        results = []
        for food in detected_foods:
            nutrient_info = f"Calories: {food['nutrients']['Energy']:.2f} kcal, Protein: {food['nutrients']['Protein']:.2f} g, Carbs: {food['nutrients']['Carbohydrate, by difference']:.2f} g, Fat: {food['nutrients']['Total lipid (fat)']:.2f} g"
            results.append(f"Detected: {food['label']}, Estimated Weight: {food['weight']:.2f} g\nEstimated Nutrients: {nutrient_info}")
        
        # Add total nutrients summary
        total_summary = f"\nTotal Nutrients:\nCalories: {total_nutrients['Energy']:.2f} kcal\nProtein: {total_nutrients['Protein']:.2f} g\nCarbs: {total_nutrients['Carbohydrate, by difference']:.2f} g\nFat: {total_nutrients['Total lipid (fat)']:.2f} g"
        results.append(total_summary)
        
        return "\n\n".join(results)
    else:
        return "No food items detected with sufficient confidence."

def recognize_and_estimate_with_focus(image, central_fraction):
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    
    focus_area = get_focus_area(image.size, central_fraction)
    nutrient_estimation = recognize_and_estimate_nutrients(image, central_fraction)
    image_with_focus = draw_focus_area(image.copy(), focus_area)
    return nutrient_estimation, image_with_focus