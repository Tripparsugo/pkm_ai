import tensorflow as tf


def get_model() -> tf.keras.models.Sequential:
    model_dir_out = "./mod_p/0/"
    cache = tf.keras.models.load_model(model_dir_out)
    return cache
