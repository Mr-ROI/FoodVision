
import os
import tensorflow as tf
import gradio as gr
from pathlib import Path
from model import create_model, preprocess_img


model = create_model()


def predict_fn(img):

  img = tf.expand_dims(preprocess_img(img), axis=0)
  pred_probs = model.predict(img)
  pred_labels_and_probs = {class_names[i]: pred_probs[0][i] for i in range(len(class_names)-1)}
  
  return pred_labels_and_probs


class_names_path = Path("class_names.txt")
with open(class_names_path, 'r') as file:
  class_names = file.read()
class_names = class_names.split("\n")[:-1]


example_list = [["examples/" + example] for example in os.listdir("examples")]

title = "FoodVision"
description = "A DeepLearning Model to identify 101 different food types"
article = "Uses the Food101 dataset and EfficientNetV2B0 model for predictions"
demo = gr.Interface(fn=predict_fn,
                    inputs=gr.Image('pil'),
                    outputs=gr.Label(num_top_classes=5, label="Prediction"),
                    title=title,
                    description=description,
                    article=article,
                    examples=example_list)


demo.launch(debug=True)
