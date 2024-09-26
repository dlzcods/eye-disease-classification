import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai 
import os

model_path = 'model'
model = tf.saved_model.load(model_path)

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

labels = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

def get_disease_detail(disease_name):
  prompt = f"Diagnosis: {disease_name}\n\nWhat is it?\n(Description about {disease_name})\n\nWhat cause it?\n(Explain what causes {disease_name})\n\nSuggestion\n(Suggestion to user)\n\nReminder: Always seek professional help, such as a doctor."
  response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
  return response.text

def predict_image(image):
  image_resized = image.resize((224, 224))
  image_array = np.array(image_resized).astype(np.float32) / 255.0
  image_array = np.expand_dims(image_array, axis=0)

  predictions = model.signatures['serving_default'](tf.convert_to_tensor(image_array, dtype=tf.float32))['output_0']
    
  # Highest prediction
  top_index = np.argmax(predictions.numpy(), axis=1)[0]
  top_label = labels[top_index]
  top_probability = predictions.numpy()[0][top_index]

  explanation = get_disease_detail(top_label)

  return {top_label: top_probability, "explanation": explanation}  

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
    outputs=[gr.Label(num_top_classes=1, label="Prediction"), gr.Textbox(label="Explanation")],
    examples=example_images,
    title="Eye Diseases Classifier",
    description="Upload an image of an eye fundus, and the model will predict it.\n\n**Disclaimer:** This model is intended as a form of learning process in the field of health-related machine learning and was trained with a limited amount and variety of data with a total of about 4000 data, so the prediction results may not always be correct. There is still a lot of room for improvisation on this model in the future.",
    allow_flagging="never"
)

interface.launch(share=True)
