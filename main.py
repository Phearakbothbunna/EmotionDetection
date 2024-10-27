import numpy as np
import os
from prepare_data import load_data, preprocess_data

def main():
    # path to train and test folders
    train_folder = './data/train'
    test_folder = './data/test'

    label_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
    # create a reverse label map to get keys (emotion names) from values (indices)
    # our team can use this later on when displaying the images
    reverse_label_map = {v: k for k, v in label_map.items()}
    # our team decided to save the preprocessed data to disk so that we don't have to rerun the load_data code everytime
    # check if preprocessed data is already saved (if not load & preprocess)
    # For train data
    if not os.path.exists('preprocessed_train_imgs.npy') or not os.path.exists('train_labels.npy'):
        print("Loading and preprocessing train data...")
        train_imgs, train_labels = load_data(train_folder)
        preprocess_train_imgs = [preprocess_data(img) for img in train_imgs]
        preprocess_train_imgs = np.array(preprocess_train_imgs, dtype=np.float32)
        train_labels = np.array([label_map[label] for label in  train_labels], dtype=np.int32)
        # Save preprocessed data for future use
        np.save('preprocessed_train_imgs.npy', preprocess_train_imgs)
        np.save('train_labels.npy', train_labels)
    else:
        # load preprocessed data directly from disk
        print("Loading preprocessed train data from disk...")
        preprocess_train_imgs = np.load('preprocessed_train_imgs.npy')
        train_labels = np.load('train_labels.npy')

    # for test data
    if not os.path.exists('preprocessed_test_imgs.npy') or not os.path.exists('test_labels.npy'):
        print("Loading and preprocessing test data...")
        test_imgs, test_labels = load_data(test_folder)
        preprocess_test_imgs = [preprocess_data(img) for img in test_imgs]
        preprocess_test_imgs = np.array(preprocess_test_imgs, dtype=np.float32)
        test_labels = np.array([label_map[label] for label in  test_labels], dtype=np.int32)
        # Save preprocessed data for future use
        np.save('preprocessed_test_imgs.npy', preprocess_test_imgs)
        np.save('test_labels.npy', test_labels)
    else:
        # load preprocessed data directly from disk
        print("Loading preprocessed test data from disk...")
        preprocess_test_imgs = np.load('preprocessed_test_imgs.npy')
        test_labels = np.load('test_labels.npy')
    return preprocess_train_imgs, train_labels, preprocess_test_imgs, test_labels


main()
