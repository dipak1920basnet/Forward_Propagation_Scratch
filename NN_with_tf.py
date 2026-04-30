from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(25, activation="relu"),
    Dense(15),
    Dropout(0.2),
    Dense(1)
]
)

# model.compile(optimizer="adam",
#               loss = BinaryCrossentropy,
#               metrics=["accuracy"]
# )

model.compile(optimizer="adam",
              loss = SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=5)

evaluation = model.evaluate(x_test, y_test)
print(evaluation)