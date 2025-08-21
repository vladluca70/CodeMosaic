import numpy as np
import tensorflow as tf

X = np.linspace(-2*np.pi, 2*np.pi, 100).reshape(-1, 1)
y = np.sin(X)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=[1], activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1)  # implicit linear
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=1000, verbose=False)

print(model.predict(np.array([0])))
print(model.predict(np.array([np.pi/2])))
print(model.predict(np.array([np.pi])))

