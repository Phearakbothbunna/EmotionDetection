import numpy as np
from sklearn.model_selection import train_test_split

from prepare_data import prepare_data
from train_model import create_cnn_model, train_model, evaluate_model, plot_images

def main():
    # Paths to the train and test folders
    train_folder = './data/train'
    test_folder = './data/test'

    # prepare data
    preprocess_train_imgs, train_labels, preprocess_test_imgs, test_labels = prepare_data(train_folder, test_folder)

    # split training data into train and validation sets
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        preprocess_train_imgs, train_labels, test_size=0.2, random_state=42)

    # add the dimensions to include the channel axis (at the end of array)
    # this ensures that all images has the shape required by the CNN model
    # new shape will be: (num_samples, h, w, 1)
    # 1 represents the single grayscale channel
    train_imgs = np.expand_dims(train_imgs, axis=-1)
    val_imgs = np.expand_dims(val_imgs, axis=-1)
    test_imgs = np.expand_dims(preprocess_test_imgs, axis=-1)

    # build the CNN model
    model = create_cnn_model()
    # train the model
    training_log = train_model(model, train_imgs, train_labels, val_imgs, val_labels)
    # evaluate the model on testing data
    evaluate_model(model, test_imgs, test_labels)
    # make predictions on the testing images
    predictions = model.predict(test_imgs)
    predicted_classes = np.argmax(predictions, axis=1)
    # visualize the testing predictions
    plot_images(test_imgs, test_labels, predicted_classes)
main()