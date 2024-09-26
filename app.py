import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai 
import os

# Load TensorFlow model
model_path = 'model'
model = tf.saved_model.load(model_path)

# Set up Gemini API
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Labels for classification
labels = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

def get_disease_detail(disease_name):
    prompt = (
        f"Diagnosis: {disease_name}\n\n"
        "What is it?\n(Description about the disease)\n\n"
        "What causes it?\n(Explain what causes the disease)\n\n"
        "Suggestions\n(Suggestion to user)\n\n"
        "Reminder: Always seek professional help, such as a doctor."
    )
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)

    # Make sure we check for candidates and handle possible missing attributes correctly
    if response.candidates and response.candidates[0].text:
        return response.candidates[0].text.strip()
    else:
        return "No detailed explanation available."

def safe_extract_section(text, start_keyword, end_keyword):
    """ Safely extract sections from the Gemini response based on start and end keywords."""
    if start_keyword in text and end_keyword in text:
        return text.split(start_keyword)[1].split(end_keyword)[0].strip()
    elif start_keyword in text:
        return text.split(start_keyword)[1].strip()
    else:
        return "Information not available."

def predict_image(image):
    # Preprocess the image
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Run prediction
    predictions = model.signatures['serving_default'](tf.convert_to_tensor(image_array, dtype=tf.float32))['output_0']
    top_index = np.argmax(predictions.numpy(), axis=1)[0]
    top_label = labels[top_index]
    top_probability = predictions.numpy()[0][top_index] * 100  # Convert to percentage

    # Get explanation from Gemini API
    explanation = get_disease_detail(top_label)

    # Extract relevant sections from the explanation
    diagnosis_section = f"**Diagnosis:** {top_label}"
    what_is_it = safe_extract_section(explanation, "What is it?", "What causes it?")
    causes = safe_extract_section(explanation, "What causes it?", "Suggestions")
    suggestions = safe_extract_section(explanation, "Suggestions", "Reminder")
    reminder = "Always seek professional help, such as a doctor."

    # Format explanation
    formatted_explanation = (
        f"{diagnosis_section}\n\n"
        f"**What is it?** {what_is_it}\n\n"
        f"**What causes it?** {causes}\n\n"
        f"**Suggestions:** {suggestions}\n\n"
        f"**Reminder:** {reminder}"
    )

    # Return both the prediction and the explanation
    return {top_label: top_probability}, formatted_explanation

# Example images
example_images = [
    ["exp_eye_images/0_right_h.png"],
    ["exp_eye_images/03fd50da928d_dr.png"],
    ["exp_eye_images/108_right_h.png"],
    ["exp_eye_images/1062_right_c.png"],
    ["exp_eye_images/1084_right_c.png"],
    ["exp_eye_images/image_1002_g.jpg"]
]

# Gradio Interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=1, label="Prediction"), 
        gr.Textbox(label="Explanation")
    ],
    examples=example_images,
    title="Eye Diseases Classifier",
    description=(
        "Upload an image of an eye fundus, and the model will predict it.\n\n"
        "**Disclaimer:** This model is intended as a form of learning process in the field of health-related machine learning and was trained with a limited amount and variety of data with a total of about 4000 data, so the prediction results may not always be correct. There is still a lot of room for improvisation on this model in the future."
    ),
    allow_flagging="never"
)

interface.launch(share=True)
