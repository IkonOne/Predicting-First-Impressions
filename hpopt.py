import model_utils
import data_prep

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras.applications import vgg16

def main():
    label_idx = 0

    X, y, labels, img_names = data_prep.load_cleaned_data(normalize=False)
    print(f'Training on label: {labels[label_idx]}')
    X = np.expand_dims(X, axis=-1)
    y = y[:,label_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(f'Target value range: {np.min(y)}, {np.max(y)}')
    X_train = data_prep.convert_greyscale_to_rgb(X_train)
    X_test = data_prep.convert_greyscale_to_rgb(X_test)
    input_shape = X_train.shape[1:]

    def build_model(hp):
        preprocessing_layers = model_utils.get_preprocessing_layers(X_train)

        vgg = vgg16.VGG16(
            include_top=False, weights=None,
            input_tensor=preprocessing_layers
        )

        hp_conv_dropout = hp.Float('conv_dropout', 0.0, 0.6)
        hp_dense_dropout = hp.Float('dense_dropout', 0.0, 0.6)
        vgg = model_utils.insert_dropout(vgg, conv_dropout=hp_conv_dropout, dense_dropout=hp_dense_dropout)
        model = keras.Sequential(vgg)

        hp_decoder_dropout = hp.Float('decoder_dropout', 0.0, 0.6)
        model = model_utils.add_value_decoder(model, [512, 128, 32], dropout=hp_decoder_dropout)

        hp_learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])
        model.compile(
            # optimizer=keras.optimizers.RMSprop(learning_rate=hp_learning_rate),
            optimizer=keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.82, beta_2=0.9),
            loss=keras.losses.MeanSquaredError(),
            metrics=[tfa.metrics.RSquare(dtype=tf.float32, y_shape=(1,))]
        )
        return model

    batch_size = 8

    tuner = kt.Hyperband(
        build_model,
        objective=kt.Objective('val_r_square', direction='max'),
        max_epochs=20,
        directory='hpopt',
        project_name=f'vgg16_{labels[label_idx]}'
    )
    tuner.search(X_train, y_train, epochs=10, validation_split=0.2, batch_size=batch_size)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print()
    print(f'Hyperparameter tuning completed...')
    for key in best_hps.values.keys():
        val = best_hps.get(key)
        print(f'Best value for {key} : {val}')
    print()

    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)

    val_acc_per_epoch = history.history['val_r_square']
    best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    history = hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_split=0.2, batch_size=batch_size)

    eval_result = hypermodel.evaluate(X_test, y_test)
    print("[test loss, test r_square]:", eval_result)

    print(history.history)

if __name__ == '__main__':
    main()