# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# from rl.agents.dqn import DQNAgent
# from rl.memory import SequentialMemory
# from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
# from tensorflow.keras.optimizers import Adam

import tensorflow as tf
from tensorflow import keras
import tensorflowjs

import json


def create_model():
    a = tf.tfjs
    model = tf.keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define a simple sequential model

    # (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    #
    # train_labels = train_labels[:1000]
    # test_labels = test_labels[:1000]
    #
    # train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    # test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
    #
    # # Create a basic model instance
    # model = create_model()
    #
    # # Display the model's architecture
    # model.summary()
    #
    # model.fit(train_images,
    #           train_labels,
    #           epochs=10,
    #           validation_data=(test_images, test_labels))

    # j = None
    # with open("./mod/0/model.json") as f:
    #     j = f.read()
    #     j = json.loads(j)
    #     j = j["modelTopology"]
    #     j = json.dumps(j)

    model_loc = "./mod/0/model.json"
    model_dir_out = "./mod_p/0/"
    tensorflowjs.converters.convert(
        ["--input_format=tfjs_layers_model", "--output_format=keras", model_loc, model_dir_out])
    # m = tf.keras.models.load_model(model_dir_out)
    # m.summary()
    # print(m.predict([[1]*983]))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
