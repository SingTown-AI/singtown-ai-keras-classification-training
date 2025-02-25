import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import os
import keras
import tensorflow as tf
from tensorflow import lite
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Export Keras to tflite micro")
parser.add_argument("--model", type=str, help="keras model path")
parser.add_argument("--dataset", type=str, help="dataset path for quantization")
parser.add_argument("--output", type=str, help="tflite output path")
args = parser.parse_args()

model = keras.models.load_model(args.model)
b, h, w, c = model.input.shape


def representative_dataset():
    for file_name in os.listdir(args.dataset):
        file_path = os.path.join(args.dataset, file_name)
        img = keras.utils.load_img(file_path, target_size=(h, w))
        array = keras.utils.img_to_array(img)
        yield [np.array([array / 127.5 - 1])]


converter = lite.TFLiteConverter.from_keras_model(model)
converter._experimental_disable_per_channel_quantization_for_dense_layers = True
converter.optimizations = [lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()


with open(args.output, "wb") as f:
    f.write(tflite_model)
