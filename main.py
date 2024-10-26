import numpy as np


def main():
    train_imgs = np.load('preprocessed_train_imgs.npy')
    test_imgs = np.load('preprocessed_test_imgs.npy')
    train_labels = np.load('train_labels.npy')
    test_labels = np.load('test_labels.npy')
    # display shapes of training & testing data for testing purposes
    print('Training data shape: ', train_imgs.shape)
    print('Training labels shape: ', train_labels.shape)

    print('Training data shape: ', test_imgs.shape)
    print('Training labels shape: ', test_labels.shape)

main()