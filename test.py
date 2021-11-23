import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats.mstats import linregress
import imageio
import cv2
import os
from clint.textui import progress
from random import uniform
import augmentation
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, Dense, Flatten, MaxPooling2D, Dropout
import numpy as np
from keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.preprocessing import image


def parse_csv(CSV):
    df = pd.read_csv(CSV, delim_whitespace=False, header=0, index_col=0)
    return df

#new load test data function

def load_test_data_into_memory(DIR, ANNO, ATTRIBUTE, normalize=True, rollaxis=True):
    #if DIR[:-1] != '/': DIR += '/'
    
    #files = os.listdir(DIR)
    annotations_list = list(parse_csv(ANNO).index)
    files = []
    for root, directories, files_list in os.walk(DIR):
        for name in files_list:
            if name in annotations_list:
                files.append(os.path.join(root, name))
    
    print('Number of files: ' + str(len(files)))
    X = []
    ImageOrder = []
    y = 0
    for image_path in progress.bar(files):
        #img = imageio.imread(DIR + image_path)
        img = imageio.imread(image_path)
        print('Image shape: ' + str(img.shape))
        if normalize: img = img.astype('float32') / 255.
        if rollaxis: img.shape = (1,150,130)
        else: img.shape = (150,130,1)
        X.append(img)
        y+=1
        ImageOrder.append(image_path)
    x = np.array(X)
    print('Loaded {} images into memory'.format(y))
    return x,ImageOrder

def normalizedf(df, trainPath):
    files = filter(lambda x: x in df.index.values, os.listdir(trainPath))
    y = []
    for image_path in progress.bar(files):
        #mu = df[ATTRIBUTE][image_path]
        mu = df['label'][image_path]
        y.append(mu)
    y = np.array(y)
    y = y - min(y)
    y = np.float32(y / max(y))

    return np.array(y)

def avg(data):
    return sum(data)/len(data)

def variance(data):
    mean = avg(data)
    v1 = (data - mean) ** 2
    return avg(v1)

def stddv(data):
    return np.sqrt(variance(data))
    
def get_Rsquared(y, predicted):
    m, b, r, p, e = linregress(y=y, x=predicted)
    r2 = r**2
    return r2

def main():
    import data_prep
    attributes = ['IQ', 'Age', 'Trustworthiness', 'Dominance']

    X, y, attribute_labels, img_names = data_prep.load_cleaned_data()

    for attribute in attributes:

        test_split = data_prep.get_premade_split(attribute, split='test')
        test_mask = np.isin(img_names, test_split)

        X_test = X[test_mask]
        y_test = y[test_mask]

        

#so we could try to just load the images from the test folder and then see what predicted outputs are generated :)
if __name__ == '__main__':
    main()
