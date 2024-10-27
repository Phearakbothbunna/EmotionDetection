import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from prepare_data import reverse_label_map

# build the cnn model
# we are using sequential model so we can add 1 layer at a time in order
def create_cnn_model():
    model = tf.keras.Sequential([
        # 1st convolution layer with 32 filters (uses ReLU activation function)
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        # 1st max pooling layer (2 x 2 pool size)
        # reduces spatial dimensions (height & width) to decrease computational load & help with overfitting
        tf.keras.layers.MaxPooling2D((2, 2)),
        # 2nd convolution layer with 64 filters
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # 2nd max pooling layer
        tf.keras.layers.MaxPooling2D((2, 2)),
        # 3rd conv layer with 128 filters
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        # 3rd max pooling
        tf.keras.layers.MaxPooling2D((2, 2)),
        # flatten 3D output from previous layer to 1D (needed for fully connected layers)
        tf.keras.layers.Flatten(),
        # fully connected layer with 128 units
        # this final dense layer uses the learned features to output probs for each of the 7 emotions
        # and the one with the highest probs is the predicted one
        tf.keras.layers.Dense(128, activation='relu'),
        # 7 classes for emotions (output layer)
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    # our team decided to use the Adam optimizer and the Sparse categorical entropy loss
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# function to train the CNN model on training and validation data
def train_model(model, train_imgs, train_labels, val_imgs, val_labels):
    # our team decided to go with 15 epochs for now to prevent overfitting but that might change later
    train_logs = model.fit(train_imgs, train_labels, epochs=15, batch_size=128, validation_data=(val_imgs, val_labels))
    # save the trained model
    model.save('emotion_cnn_model.keras')
    # return the object that has info about the training process
    # it keeps track of loss value, accuracy, validation and validation accuracy
    return train_logs

# For testing data: evaluate CNN model's performance
def evaluate_model(model, test_imgs, test_labels):
    test_loss, test_accuracy = model.evaluate(test_imgs, test_labels)
    # print the test accuracy
    print(f'Test accuracy: {test_accuracy:.2f}')

# Display the testing images (20 of them) with the true & predicted labels
def plot_images(images, labels, predicted_labels=None):
    # randomly select 20 indices to get random images
    rand_indices = np.random.choice(len(images), 20, replace=False)
    plt.figure(figsize=(12, 12))
    # i help control where each subplot is placed in the 4 x 5 grid
    # idx is the random index of the image
    for i, idx in enumerate(rand_indices):
        # 4 x 5 grid
        plt.subplot(4, 5, i + 1)
        # display images in grayscale
        plt.imshow(images[idx].reshape(48, 48), cmap='gray')
        true_label = reverse_label_map[labels[rand_indices[i]]]
        title = f'True: {true_label}'
        if predicted_labels is not None:
            predicted_label = reverse_label_map[predicted_labels[rand_indices[i]]]
            title += f'\nPred: {predicted_label}'
        plt.title(title)
        plt.axis('off')
    plt.show()


