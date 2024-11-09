import numpy as np
from sklearn.model_selection import train_test_split
from prepare_data import prepare_data
from train_model import create_cnn_model, train_model, evaluate_model, plot_images

def main():
    # Paths to the train and test folders
    train_folder = './data/train'
    test_folder = './data/test'

    # Prepare data
    preprocess_train_imgs, train_labels, preprocess_test_imgs, test_labels = prepare_data(train_folder, test_folder)

    # Split training data into train and validation sets
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        preprocess_train_imgs, train_labels, test_size=0.2, random_state=42)

    # Add the channel axis (grayscale images have one channel)
    train_imgs = np.expand_dims(train_imgs, axis=-1)
    val_imgs = np.expand_dims(val_imgs, axis=-1)
    test_imgs = np.expand_dims(preprocess_test_imgs, axis=-1)

    # Build and train the CNN model
    model = create_cnn_model()
    training_log = train_model(model, train_imgs, train_labels, val_imgs, val_labels)

    # Evaluate the model on test data
    evaluate_model(model, test_imgs, test_labels)

    # Make predictions on the test set
    predictions = model.predict(test_imgs)
    predicted_classes = np.argmax(predictions, axis=1)

    # Visualize predictions on the test set
    plot_images(test_imgs, test_labels, predicted_classes)

if __name__ == "__main__":
    main()
