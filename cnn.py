import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import lime
import lime.lime_image
input_shape = (224, 224, 3)
num_classes = 4
batch_size = 32
learning_rate = 0.001
epochs = 3
wsi_dir = "G:/Photos"
data_generator = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
train_dataset = data_generator.flow_from_directory(
    wsi_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    subset="training"
)
val_dataset = data_generator.flow_from_directory(
    wsi_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    subset="validation"
)
model = keras.Sequential([
    ResNet50(include_top=False, weights="imagenet", input_shape=input_shape),
    layers.GlobalAveragePooling2D(),
    layers.Dense(num_classes, activation="softmax")
])
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)
img, label = val_dataset.next()
img = img[0]
img_array = np.expand_dims(img, axis=0)
explainer = lime.lime_image.LimeImageExplainer()
preds = model.predict(img_array)
pred_class = np.argmax(preds)
import lime
from lime import lime_image
explainer = lime_image.LimeImageExplainer()
def explain_image(img):
    explanation = explainer.explain_instance(img, model.predict, top_labels=1, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=True)
    return temp
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
