import os
import numpy as np
import pandas as pd
import cv2

# (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
def load_data(data_folder):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    images = []
    labels = []
    # there're 7 different emotions folders, so we go through each of them
    # ex: emotion_dir = train/happy/
    for emotion in emotions:
        emotion_dir = os.path.join(data_folder, emotion)
        # go through the images in for each emotion folder
        for img_dir in os.listdir(emotion_dir):
            img = cv2.imread(os.path.join(emotion_dir, img_dir), cv2.IMREAD_GRAYSCALE)
            # if image can be read, append them to the image list
            if img is not None:
                images.append(img)
                labels.append(emotion)
    return images, labels

def preprocess_data(image):
    # The KER2013 dataset already has images that are 48x48 and in grayscale
    # Normalize pixel values (to get a range between 0 and 1)
    normalize_img = image / 255.0
    return normalize_img

# path to train and test folders
train_folder = './data/train'
test_folder = './data/test'
# our team decided to save the preprocessed data to disk so that we don't have to rerun the load_data code everytime
# check if preprocessed data is already saved (if not load & preprocess)
# For train data
if not os.path.exists('preprocessed_train_imgs.npy') or not os.path.exists('train_labels.npy'):
    print("Loading and preprocessing train data...")
    train_imgs, train_labels = load_data(train_folder)
    preprocess_train_imgs = [preprocess_data(img) for img in train_imgs]
    preprocess_train_imgs = np.array(preprocess_train_imgs)
    train_labels = np.array(train_labels)
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
    preprocess_test_imgs = np.array(preprocess_test_imgs)
    test_labels = np.array(test_labels)
    # Save preprocessed data for future use
    np.save('preprocessed_test_imgs.npy', preprocess_test_imgs)
    np.save('test_labels.npy', test_labels)
else:
    # load preprocessed data directly from disk
    print("Loading preprocessed test data from disk...")
    preprocess_test_imgs = np.load('preprocessed_test_imgs.npy')
    train_labels = np.load('test_labels.npy')

