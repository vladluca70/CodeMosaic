import numpy as np
import tensorflow as tf

x=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
y=2*x+1

model=tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x,y, epochs=400, verbose=False)

print(model.predict(np.array([8.0])))
print(model.predict(np.array([9.0])))
print(model.predict(np.array([10.0])))
print(model.predict(np.array([11.0])))
print(model.predict(np.array([5.0])))

