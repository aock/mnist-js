# train and save MNIST model on keras for tensorflowjs
# $ python3 -m venv tfjs
# $ ./tfjs/bin/pip install tensorflowjs
# $ ./tfjs/bin/python keras-mnist.py
from tensorflow import keras
from keras import models as m
from keras import layers as l
from keras import datasets as d
from keras import utils as u
import tensorflowjs as tfjs

# categorizing model
model = m.Sequential()
model.add(l.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(l.Conv2D(64, (3, 3), activation="relu"))
model.add(l.MaxPooling2D(pool_size=(2,2)))
model.add(l.Dropout(0.25))
model.add(l.Flatten())
model.add(l.Dense(128, activation="relu"))
model.add(l.Dropout(0.5))
model.add(l.Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adadelta",
              metrics=["accuracy"])

# MNIST data
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = d.mnist.load_data()
x_train = x_train_raw.reshape(*x_train_raw.shape, 1).astype("float32") / 255;
x_test = x_test_raw.reshape(*x_test_raw.shape, 1).astype("float32") / 255;
y_train = u.to_categorical(y_train_raw, 10)
y_test = u.to_categorical(y_test_raw, 10)

# train
history = model.fit(x_train, y_train, batch_size=128, epochs=12, verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test)
print("[score] {}, accuracy: {}".format(*score))

#model.save("mnist-model.h5")
tfjs.converters.save_keras_model(model, ".")
