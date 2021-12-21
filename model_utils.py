import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_preprocessing_layers(X_train, normalization=True):
    preprocessing_layers = layers.Input(name='input', shape=X_train.shape[1:])
    if normalization:
        norm = layers.Normalization(axis=None)
        norm.adapt(X_train)
        preprocessing_layers = norm(preprocessing_layers)

    return preprocessing_layers

def insert_dropout(model, conv_dropout=0.2, dense_dropout=0.5):
    """Inserts dropouts after convolution and dense layers in an existing keras.Sequential model"""
    x = model.layers[0].output
    for i in range(1, len(model.layers)-1):
        layer = model.layers[i]
        x = layer(x)
        if conv_dropout != 0.0 and isinstance(layer, layers.Conv2D):
            x = layers.Dropout(conv_dropout)(x)
        elif dense_dropout != 0.0 and isinstance(layer, layers.Dense):
            x = layers.Dropout(dense_dropout)(x)
    
    x = model.layers[-1](x)
    
    return keras.Model(inputs=model.input, outputs=x)

def add_value_decoder(sequential, sizes, dropout=0.2):
    sequential.add(layers.Conv2D(512, (3,3), activation='relu'))
    sequential.add(layers.Flatten())
    for size in sizes:
        sequential.add(layers.Dense(size))
        sequential.add(layers.LeakyReLU(alpha=0.2))
        if dropout != 0.0:
            sequential.add(layers.Dropout(dropout))
    sequential.add(layers.Dense(1, activation='linear'))
    return sequential

if __name__ == '__main__':
    import data_prep
    X_train, X_test, y_train, y_test = data_prep.load_prepped_data()

    vgg = keras.applications.vgg16.VGG16(
        include_top=True, weights=None,
        input_tensor=get_preprocessing_layers(X_train)
    )
    model = insert_dropout(vgg)
    print(model.summary())