import tensorflow as tf
from pathlib import Path
from PIL import Image

def create_model():
  model_path = "FineTunedModelFood101.keras"
  path_to_model = Path(model_path)
  model = tf.keras.models.load_model(path_to_model)

  return model

def preprocess_img(img, image_shape:int=224):

  try:
    img = Image.open(img)
  except:
    pass
  img = tf.image.resize(img, [image_shape, image_shape])
  img = tf.cast(img, dtype=tf.float32)

  return img
