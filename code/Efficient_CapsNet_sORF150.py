import numpy as np
import tensorflow as tf
from layers import PrimaryCaps, FCCaps, Length, Mask


def efficient_capsnet_sORF(input_shape):
    """
        Efficient-CapsNet graph architecture.

        Parameters
        ----------
        input_shape: list
            network input shape
    """
    inputs = tf.keras.Input(input_shape)

    x = tf.keras.layers.Conv2D(64, 1, activation='relu', padding='valid', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 2, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, 3, 2, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = PrimaryCaps(128, 4, 64, 6)(x) #


    digit_caps = FCCaps(2, 64)(x) #

    digit_caps_len = Length(name='Length_capsnet_output')(digit_caps)

    return tf.keras.Model(inputs=inputs, outputs=[digit_caps, digit_caps_len], name='Efficient_CapsNet')


def generator_sORF(input_shape):
    """
        Generator graph architecture.

        Parameters
        ----------
        input_shape: list
            network input shape
    """
    inputs = tf.keras.Input(64*2)  #

    # x = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid', kernel_initializer='glorot_normal')(x)
    x = tf.keras.layers.Reshape(target_shape=input_shape, name='out_generator')(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name='Generator')


def build_sORF(input_shape, mode, verbose):
    """
        Efficient-CapsNet graph architecture with reconstruction regularizer. The network can be initialize with different modalities.

        Parameters
        ----------
        input_shape: list
            network input shape
        mode: str
            working mode ('train', 'test' & 'play')
        verbose: bool
    """
    inputs = tf.keras.Input(input_shape)
    y_true = tf.keras.layers.Input(shape=(2,))
    noise = tf.keras.layers.Input(shape=(2, 64))   #

    efficient_capsnet = efficient_capsnet_sORF(input_shape)

    if verbose:
        efficient_capsnet.summary()
        print("\n\n")
    digit_caps, digit_caps_len = efficient_capsnet(inputs)
    noise_digitcaps = tf.keras.layers.Add()([digit_caps, noise]) #only if mode is play

    masked_by_y = Mask()([digit_caps, y_true])
    masked = Mask()(digit_caps)
    masked_noised_y = Mask()([noise_digitcaps, y_true])

    generator = generator_sORF(input_shape)

    if verbose:
        generator.summary()
        print("\n\n")

    x_gen_train = generator(masked_by_y)
    x_gen_eval = generator(masked)
    x_gen_play = generator(masked_noised_y)

    if mode == 'train':
        return tf.keras.models.Model([inputs, y_true], [digit_caps_len, x_gen_train], name='Efficient_CapsNet_Generator')
    elif mode == 'test':
        return tf.keras.models.Model(inputs, [digit_caps_len, x_gen_eval], name='Efficient_CapsNet_Generator')
    elif mode == 'play':
        return tf.keras.models.Model([inputs, y_true, noise], [digit_caps_len, x_gen_play], name='Efficient_CapsNet_Generator')
    else:
        raise RuntimeError('mode not recognized')

    # train_model = tf.keras.models.Model([inputs, y_true], [digit_caps_len, x_gen_train], name='Efficient_CapsNet_Generatortrain')
    # eval_model = tf.keras.models.Model(inputs, [digit_caps_len, x_gen_eval], name='Efficient_CapsNet_Generatortest')
    # play_model = tf.keras.models.Model([inputs, y_true, noise], [digit_caps_len, x_gen_play], name='Efficient_CapsNet_Generatorplay')
    # return train_model, eval_model, play_model









