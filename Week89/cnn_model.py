# model.py

import tensorflow as tf

def cnn_model(input_shape=(267, 9, 1), num_classes=128):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(10, (16, 2), activation='relu', padding='valid', input_shape=input_shape, name="conv1"),
        tf.keras.layers.MaxPooling2D((2, 2), name="pool1"),
        tf.keras.layers.Conv2D(20, (11, 3), activation='relu', padding='valid', name="conv2"),
        tf.keras.layers.MaxPooling2D((2, 2), name="pool2"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(256, activation='relu', name="fc1"),
        tf.keras.layers.Dropout(0.5, name="dropout"),
        tf.keras.layers.Dense(num_classes, activation='sigmoid', name="fc2")
    ])
    return model
