import numpy as np
import tensorflow as tf

X = np.random.randn(300, 2)   
y = (X[:,0] + X[:,1] >= 0).astype(int)

model=tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[2], activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X,y,epochs=500, verbose=False)

p1=model.predict(np.array([[1.23434, 0.34534]]))
p2=model.predict(np.array([[-0.53353, 1.43564]]))
p3=model.predict(np.array([[-0.343453, 0.12432]]))
p4=model.predict(np.array([[-1.32435, -0.23435]]))

if p1>=0.5:
    print("1")
else:
    print('0')

if p2>=0.5:
    print("1")
else:
    print('0')

if p3>=0.5:
    print("1")
else:
    print('0')

if p4>=0.5:
    print("1")
else:
    print('0')
