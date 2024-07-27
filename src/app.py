import gradio as gr
from food_recognition import recognize_and_estimate_nutrients

def main():
    interface = gr.Interface(
        fn=recognize_and_estimate_nutrients,
        inputs=gr.Image(type="pil", label="Upload Image"),
        outputs="text",
        title="Nutrient and Calorie Estimator",
        description="Upload an image of a food item. The model will detect the food, estimate its weight, and estimate its nutrients and calories."
    )
    interface.launch()

if __name__ == '__main__':
    main()
