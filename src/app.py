import gradio as gr
from food_recognition import recognize_and_estimate_with_focus

def main():
    interface = gr.Interface(
        fn=recognize_and_estimate_with_focus,
        inputs=[
            gr.Image(type="pil", label="Upload Image"),
            gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.7, label="Focus Area Fraction")
        ],
        outputs=[
            gr.Textbox(label="Nutrient Estimation"),
            gr.Image(type="pil", label="Image with Focus Area")
        ],
        title="Nutrient and Calorie Estimator",
        description="Upload an image of a food item. Adjust the focus area fraction to indicate the central focus area for food detection. The model will detect the food, estimate its weight, and calculate its nutrients and calories."
    )
    interface.launch()

if __name__ == '__main__':
    main()