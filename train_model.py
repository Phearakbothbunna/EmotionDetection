import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


# load the saved preprocessed data
preprocess_train_imgs = np.load('preprocessed_train_imgs.npy')
train_labels = np.load('train_labels.npy')
preprocess_test_imgs = np.load('preprocessed_test_imgs.npy')
test_labels = np.load('test_labels.npy')

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
model = create_cnn_model()

# function to train the CNN model on training and validation data
def train_model(model, train_imgs, train_labels, val_imgs, val_labels):
    # our team decided to go with 15 epochs for now to prevent overfitting but that might change later
    train_logs = model.fit(train_imgs, train_labels, epochs=15, batch_size=128, validation_data=(val_imgs, val_labels))
    # save the trained model
    model.save('emotion_cnn_model.keras')
    # return the object that has info about the training process
    # it keeps track of loss value, accuracy, validation and validation accuracy
    return train_logs
# train the model
train_logs = train_model(model, train_imgs, train_labels, val_imgs, val_labels)

# For testing data: evaluate CNN model's performance
def evaluate_model(model, test_imgs, test_labels):
    test_loss, test_accuracy = model.evaluate(test_imgs, test_labels)
    # print the test accuracy
    print(f'Test accuracy: {test_accuracy:.2f}')
evaluate_model(model, test_imgs, test_labels)

# Display the testing images (10 of them) with the true & predicted labels
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
        plt.title(f'True: {labels[idx]}' + (f'\nPred: {predicted_labels[idx]}' if predicted_labels is not None else ''))
        plt.axis('off')
    plt.show()
# make predictions on the testing images
predictions = model.predict(test_imgs)
# convert prediction probabilities to class labels
predicted_classes = np.argmax(predictions, axis=1)
# visualize the testing predictions
plot_images(test_imgs, test_labels, predicted_classes)
