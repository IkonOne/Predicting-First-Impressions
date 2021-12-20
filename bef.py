import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications import xception, mobilenet
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split

class BEFModel(keras.Model):
    def __init__(self, black_box) -> None:
        """Creates a Bayesian Evaluation Framework model.

        Keyword arguments:
        black_box -- The black box encoder model.
        """
        super().__init__()
        self.bb = black_box
    
def RSquared(y_true, y_pred):
    from tensorflow.keras import backend as K
    RSS = K.sum(K.square(y_true - y_pred))
    TSS = K.maximum(
        K.sum(K.square(y_true - K.mean(y_true))), 
        K.epsilon() # don't divide by 0
    )
    return 1 - RSS/TSS

def RSquared_inv(y_true, y_pred):
    return -RSquared(y_true, y_pred)

def main():
    import data_prep

    label_idx = 2

    X, y, labels, img_names = data_prep.load_cleaned_data(normalize=False)
    print(f'Training on label: {labels[label_idx]}')
    X = np.expand_dims(X, axis=-1)
    y = y[:,label_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(f'Target value range: {np.min(y)}, {np.max(y)}')
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_train = tf.image.grayscale_to_rgb(X_train)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    X_test = tf.image.grayscale_to_rgb(X_test)
    input_shape = X_train.shape[1:]

    preprocessing_layers = layers.Input(name='input', shape=input_shape)
    norm = layers.Normalization(axis=None)
    norm.adapt(X_train)
    preprocessing_layers = norm(preprocessing_layers)

    mob = mobilenet.MobileNet(
        include_top=False, weights=None,
        input_tensor=preprocessing_layers
    )

    model = keras.Sequential(mob)
    model.add(layers.Flatten())
    model.add(layers.Dense(2048))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dense(32))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dense(1))
    print(model.summary())

    batch_size = 32
    nepochs = 50

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=pow(10, -4.6)),
        loss=keras.losses.MeanSquaredError(),
        metrics=[tfa.metrics.RSquare(dtype=tf.float32, y_shape=(1,))]
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=nepochs
    )
    print(history.history)
    # bef = BEFModel(bb)

if __name__ == '__main__':
    main()
    