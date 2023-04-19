from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Sequential
import time
from tensorflow.python.client import device_lib


def create_model():
    input_shape = (1, 300)
    model = Sequential()
    model.add(Dense(200, activation="elu", input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(50, activation="elu"))
    model.add(Dense(20, activation="linear"))
    model.compile()
    return model


if __name__ == '__main__':
    print(device_lib.list_local_devices())
    model = create_model()
    inp = [[[0.5] * 300]]

    # warmup
    for i in range(0, 100):
        model.predict(inp)

    start_time = time.time()
    for i in range(0, 100):
        model.predict(inp)

    end_time = time.time()
    print(f" {end_time - start_time} s")
