import keras.models
import tensorflow as tf
from keras import models,layers
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


if __name__ == '__main__':
    loaded_model = keras.models.load_model('trained')
    y=loaded_model.predict(tf.expand_dims(x_test[0],axis=0))
    print(y_test[0])
    print(y)
    print(y.shape)
    print(y.argmax())