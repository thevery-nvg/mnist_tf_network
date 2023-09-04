import keras.models
import tensorflow as tf
from keras import models,layers

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(y_test)


model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', # (3,3) - фильтр
                        input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)), # фильтр (2,2) для пулинга
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, 'relu'),
    layers.Dense(10)
])
# Для каждого примера модель возвращает вектор оценок логитов или логарифмических шансов
# по одному для каждого класса.
predictions = model(x_train[:1]).numpy()

# преобразует эти логиты в вероятности для каждого класса
tf.nn.softmax(predictions).numpy()

# Определяем функцию потерь
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.save('trained')

