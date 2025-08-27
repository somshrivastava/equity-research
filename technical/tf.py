import numpy as np
import tensorflow as tf

# Simple test model
m = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),   # <--- define input shape
    tf.keras.layers.Dense(1)
])
m.compile(optimizer="adam", loss="mse")

# Dummy training data (must be numpy arrays)
X = np.array([[1, 2, 3]], dtype=np.float32)
y = np.array([[1]], dtype=np.float32)

m.fit(X, y, epochs=1)
