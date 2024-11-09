import os
import cv2
import numpy as np

# Emotion label mapping (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
label_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
reverse_label_map = {v: k for k, v in label_map.items()}

# Load the images and labels from the dataset folder
def load_data(data_folder):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    images = []
    labels = []
    for emotion in emotions:
        emotion_dir = os.path.join(data_folder, emotion)
        for img_dir in os.listdir(emotion_dir):
            img = cv2.imread(os.path.join(emotion_dir, img_dir), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(emotion)
    return images, labels

def preprocess_data(image):
    # Normalize the image (scale the pixel values to [0, 1])
    return image / 255.0

def save_preprocessed_data(data, labels, data_file, label_file):
    np.save(data_file, data)
    np.save(label_file, labels)

def load_preprocessed_data(data_file, label_file):
    data = np.load(data_file)
    labels = np.load(label_file)
    return data, labels

def prepare_data(train_folder, test_folder):
    # Load and preprocess training data
    if not os.path.exists('preprocessed_train_imgs.npy') or not os.path.exists('train_labels.npy'):
        print("Loading and preprocessing train data...")
        train_imgs, train_labels = load_data(train_folder)
        preprocess_train_imgs = [preprocess_data(img) for img in train_imgs]
        preprocess_train_imgs = np.array(preprocess_train_imgs, dtype=np.float32)
        train_labels = np.array([label_map[label] for label in train_labels], dtype=np.int32)
        save_preprocessed_data(preprocess_train_imgs, train_labels, 'preprocessed_train_imgs.npy', 'train_labels.npy')
    else:
        print("Loading preprocessed train data from disk...")
        preprocess_train_imgs, train_labels = load_preprocessed_data('preprocessed_train_imgs.npy', 'train_labels.npy')

    # Load and preprocess testing data
    if not os.path.exists('preprocessed_test_imgs.npy') or not os.path.exists('test_labels.npy'):
        print("Loading and preprocessing test data...")
        test_imgs, test_labels = load_data(test_folder)
        preprocess_test_imgs = [preprocess_data(img) for img in test_imgs]
        preprocess_test_imgs = np.array(preprocess_test_imgs, dtype=np.float32)
        test_labels = np.array([label_map[label] for label in test_labels], dtype=np.int32)
        save_preprocessed_data(preprocess_test_imgs, test_labels, 'preprocessed_test_imgs.npy', 'test_labels.npy')
    else:
        print("Loading preprocessed test data from disk...")
        preprocess_test_imgs, test_labels = load_preprocessed_data('preprocessed_test_imgs.npy', 'test_labels.npy')

    return preprocess_train_imgs, train_labels, preprocess_test_imgs, test_labels
