import os
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage import io
from sklearn.model_selection import train_test_split

def get_image_names():
    img_names = os.listdir('./Images')
    return img_names

def load_images(normalize=False):
    img_names = os.listdir('./Images')
    data_dir = os.path.join(os.getcwd(), 'Images')
    X = []
    for img in img_names:
        img_path = os.path.join(data_dir, img)
        X.append(io.imread(img_path))
    
    # Mean center and normalize the images
    if normalize:
        X = (X - np.mean(X)) / np.std(X)
    
    print(f'Loaded {len(X)} images...')
    
    return np.asarray(X), img_names

def convert_greyscale_to_rgb(X):
    if X.shape[-1] != 1:
        X = np.expand_dims(X, axis=-1)
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    X = tf.image.grayscale_to_rgb(X)
    return X

def load_annotations():
    annotation_labels= os.listdir('./Annotations')
    data_dir = os.path.join(os.getcwd(), 'Annotations')
    data_frames = []
    for annotation in annotation_labels:
        annotation_path = os.path.join(data_dir, annotation, 'annotations.csv')
        csv = pd.read_csv(annotation_path)
        csv = csv.rename(columns={csv.columns[0]: "Image"})
        data_frames.append(csv)
    
    data_frame = data_frames[0]
    for i in range(1, len(data_frames)):
        data_frame = pd.merge(data_frame, data_frames[i], how='outer')
    
    print(f'The annotations reference {data_frame.shape[0]} images...')

    return data_frame, annotation_labels

def load_cleaned_data(normalize=False):
    """There are images listed in the annotations that don't exist.
    So let's get rid of them and return a clean dataset."""
    X, img_names = load_images(normalize)
    annotations, annotation_labels = load_annotations()
    annotations = annotations.dropna()
    mask = np.isin(img_names, annotations['Image'])

    print(f'We are deleting {len(mask) - np.sum(mask)} images...')
    print(f'Leaving a total of {np.sum(mask)} images...')

    return X[mask], np.asarray(annotations[annotation_labels]), annotation_labels, np.asarray(annotations['Image'])

def get_premade_split(attribute, split='train', save_folder='Splits'):
    imgs = []
    with open(f'./{save_folder}/{attribute}/{split}.txt', 'r') as txtFile:
        for line in txtFile:
            imgs.append(line.rstrip())

    return imgs

def create_splits(annotation, splits=[0.9, 0.05, 0.05], save_folder='Splits', rseed=None):
    img_names = get_image_names()

    train_split,img_remaining, _,_ = train_test_split(
        img_names, img_names, train_size=splits[0], shuffle=True, random_state=rseed
    )
    
    split_size = splits[1] / np.sum(splits[1:])
    validation_split,test_split, _,_ = train_test_split(
        img_remaining, img_remaining, train_size=split_size, shuffle=True, random_state=rseed
    )

    def write_split(name, split):
        import os
        os.makedirs(f'./{save_folder}/{annotation}', exist_ok=True)
        with open(f'./{save_folder}/{annotation}/{name}.txt', 'w') as txtFile:
            txtFile.seek(0)
            for img in split:
                txtFile.write(f'{img}\n')
            txtFile.truncate()
    
    write_split('train', train_split)
    write_split('validation', validation_split)
    write_split('test', test_split)

    print(f'Created training split of ratio {splits[0]} with {len(train_split)} images')
    print(f'Created validation split of ratio {splits[1]} with {len(validation_split)} images')
    print(f'Created test split of ratio {splits[2]} with {len(test_split)} images')

def load_prepped_data(train_size=0.8, label_idx=2):
    X, y, labels, img_names = load_cleaned_data(normalize=False)
    print(f'Training on label: {labels[label_idx]}')
    X = np.expand_dims(X, axis=-1)
    y = y[:,label_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

    print(f'Target value range: {np.min(y)}, {np.max(y)}')
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_train = tf.image.grayscale_to_rgb(X_train)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    X_test = tf.image.grayscale_to_rgb(X_test)

    return X_train, X_test, y_train, y_test

def normalize_splits(train, *argv):
    """Normalizes image splits in place.

    Computes the mean and std strictly on the training data.

    """
    mu_t = np.mean(train)
    std_t = np.std(train)

    train = (train - mu_t) / std_t
    for arg in argv:
        arg = (arg - mu_t) / std_t


if __name__ == '__main__':
    # images, annotations, annotation_labels = load_cleaned_data()

    # annotations = ['IQ', 'Age', 'Trustworthiness', 'Dominance']
    # for annotation in annotations:
    #     create_splits(annotation)

    test_split = get_premade_split('IQ', split='test')
