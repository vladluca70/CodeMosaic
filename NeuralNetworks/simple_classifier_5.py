import numpy as np
import tensorflow as tf

x = np.linspace(1, 10, 100)
y = (x > 5).astype(int)  # dacă x > 5 → 1, altfel 0

model=tf.keras.Sequential([
    tf.keras.layers.Dense(units=32, activation='relu', input_shape=[1]),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(x,y, epochs=300, verbose=False)

print(model.predict(np.array([1.4])))
print(model.predict(np.array([4.7])))
print(model.predict(np.array([7.4])))
print(model.predict(np.array([5.3])))

