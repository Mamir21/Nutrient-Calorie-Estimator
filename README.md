# Nutrient and Calorie Estimator

The Nutrient and Calorie Estimator is an AI-powered application designed to help users identify and analyze the nutritional content of food items from images. By utilizing state-of-the-art object detection models and integrating with the USDA's food database, this tool provides detailed information about the calories, proteins, carbohydrates, and fats present in the detected food items.

## Demo
![1722289225999](https://github.com/user-attachments/assets/490e8bb4-72e9-4d61-ae9f-3b7a4c126bad)

## Features
- **Food Detection**: Uses the `facebook/detr-resnet-50` object detection model to identify various food items in an uploaded image.
- **Nutrient Estimation**: Estimates the nutritional content (calories, proteins, carbohydrates, and fats) of the detected food items using data from the USDA food database.
- **Focus Area Adjustment**: Allows users to adjust the focus area to ensure accurate detection of the main food items in the image.
- **Interactive Interface**: Features a user-friendly interface built with Gradio for easy interaction and visualization.

## Technologies Used
- **Python**: Backend logic and model integration.
- **Gradio**: Web-based frontend.
- **PyTorch**: Object detection model.
- **Transformers**: Implementations of transformer models.
- **Pillow**: Image processing.
- **Requests**: Fetch data from the USDA API.
- **Python-dotenv**: Load environment variables from a .env file.
- **USDA Food Database**: Source of nutritional data.

## How It Works
1. **Image Upload**: Users upload an image of the food item(s).
2. **Food Detection**: The application detects food items within the image.
3. **Focus Adjustment**: Users can adjust the focus area to improve detection accuracy.
4. **Nutrient Estimation**: The application fetches nutritional information from the USDA database and estimates the nutrient content.
5. **Result Display**: The application displays detected food items, their estimated weights, and their nutritional content, along with a total summary.

## Usage
1. **Clone the Repository**: Clone the project repository using the following command:
   ```bash
   git clone https://github.com/Farzin312/nutrient-calorie-estimator.git

2. **Navigate to the project directory**
   ```bash
   git clone https://github.com/Farzin312/nutrient-calorie-estimator.git
   
3. **Install the requirements.txt**
   ```bash
   cd nutrient-calorie-estimator
   
4. **Run the application**
   ```bash
   python src/app.py
   
## Contributors
- **Farzin Shifat**
- **Muhammad Amir**
- **MIR SHAHIDUZZAMAN**
- **Ujjwala Pothula**

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
