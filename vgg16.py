import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications import vgg16
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    import data_prep
    import model_utils 

    label_idx = 0

    X, y, labels, img_names = data_prep.load_cleaned_data(normalize=False)

    # debug using a small subset of data
    # indices = np.arange(0, len(X))
    # np.random.shuffle(indices)
    # X = X[indices[0:256]]
    # y = y[indices[0:256]]
    # print(X.shape)

    print(f'Training on label: {labels[label_idx]}')
    X = np.expand_dims(X, axis=-1)
    y = y[:,label_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)
    data_prep.normalize_splits(X_train, X_test, X_val)
    # mu_xt = np.mean(X_train)
    # std_xt = np.std(X_train)
    # X_train = (X_train - mu_xt) / std_xt
    # X_test = (X_test - mu_xt) / std_xt

    # print(f'Target value range: {np.min(y)}, {np.max(y)}')
    # X_train = data_prep.convert_greyscale_to_rgb(X_train)
    # X_val = data_prep.convert_greyscale_to_rgb(X_val)
    # X_test = data_prep.convert_greyscale_to_rgb(X_test)

    preprocessing_layers = model_utils.get_preprocessing_layers(X_train, normalization=True)

    vgg = vgg16.VGG16(
        include_top=False, weights=None,
        input_tensor=preprocessing_layers
    )

    vgg = model_utils.insert_dropout(vgg, conv_dropout=0.0, dense_dropout=0.0)
    model = keras.Sequential(vgg)
    model = model_utils.add_value_decoder(model, [2048, 512, 32], dropout=0.0)

    batch_size = 4
    nepochs = 50

    model.compile(
        # Adam values are hand tuned on random subset of data (n=255) in order [lr, b1, b2]
        # values were twiddled until the 'loss' during training appeared to make monotonic improvements
        optimizer=keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.82, beta_2=0.9),
        loss=keras.losses.MeanSquaredError(),
        metrics=[tfa.metrics.RSquare(dtype=tf.float32, y_shape=(1,))],
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=nepochs,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_r_square', patience=8, mode='max')]
    )

    loss, r_square = model.evaluate(X_test, y_test)
    print("[test loss, test r_square]:", (loss, r_square))

    model.save(f'./Models/vgg16_{labels[label_idx]}.h5')

if __name__ == '__main__':
    main()
    