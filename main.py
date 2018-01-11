import numpy as np
from keras.layers import Dense
from keras.models import Sequential

NUM_DIGITS = 10
FORMAT_STRING = "0" + str(NUM_DIGITS) + "b"


def int_to_bin(n: int):
    return np.asarray([int(x) for x in format(n, FORMAT_STRING)])


def int_to_fizz(x: int):
    if x % 5 == x % 3 == 0:
        return np.asarray((1, 0, 0, 0))
    elif x % 5 == 0:
        return np.asarray((0, 1, 0, 0))
    elif x % 3 == 0:
        return np.asarray((0, 0, 1, 0))
    else:
        return np.asarray((0, 0, 0, 1))


def evaluate(n: int):
    prediction = model.predict(np.asarray([int_to_bin(n)]))
    prediction[prediction >= 0.5] = 1
    prediction[prediction < 0.5] = 0
    return prediction


def fizz_buzz(i: int, pred: int):
    return ["fizzbuzz", "buzz", "fizz", str(i)][pred]


model = Sequential([
    Dense(32, activation="relu", input_shape=(NUM_DIGITS,)),
    Dense(4, activation="softmax")
])

y_values = np.asarray([int_to_fizz(x) for x in range(101, 2 ** NUM_DIGITS)])
x_values = np.asarray([int_to_bin(x) for x in range(101, 2 ** NUM_DIGITS)])

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_values, y_values, epochs=10000)

nums = np.arange(1, 100)
predictions = model.predict(np.asarray(list(map(int_to_bin, nums))))
output = np.vectorize(fizz_buzz)(nums, predictions.argmax(axis=1))
print(output)
