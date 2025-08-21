import numpy as np
import tensorflow as tf

x=np.random.uniform(1,30, 100)
y=x**2+3*x-1

model=tf.keras.Sequential([
    tf.keras.layers.Dense(units=32, activation='relu', input_shape=[1]),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x,y, epochs=1500, verbose=False)

print(model.predict(np.array([8.0])))
print(model.predict(np.array([9.0])))
print(model.predict(np.array([10.0])))
print(model.predict(np.array([11.0])))
print(model.predict(np.array([5.0])))

