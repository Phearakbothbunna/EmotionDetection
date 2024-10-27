import os
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



