import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from prepare_data import reverse_label_map

# Build the CNN model
def create_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train the model
def train_model(model, train_imgs, train_labels, val_imgs, val_labels):
    train_logs = model.fit(train_imgs, train_labels, epochs=15, batch_size=128, validation_data=(val_imgs, val_labels))
    model.save('emotion_cnn_model.keras')
    return train_logs

# Evaluate the model on test data
def evaluate_model(model, test_imgs, test_labels):
    test_loss, test_accuracy = model.evaluate(test_imgs, test_labels)
    print(f'Test accuracy: {test_accuracy:.2f}')

# Display images with true and predicted labels
def plot_images(images, labels, predicted_labels=None):
    rand_indices = np.random.choice(len(images), 20, replace=False)
    plt.figure(figsize=(12, 12))
    for i, idx in enumerate(rand_indices):
        plt.subplot(4, 5, i + 1)
        plt.imshow(images[idx].reshape(48, 48), cmap='gray')
        true_label = reverse_label_map[labels[rand_indices[i]]]
        title = f'True: {true_label}'
        if predicted_labels is not None:
            predicted_label = reverse_label_map[predicted_labels[rand_indices[i]]]
            title += f'\nPred: {predicted_label}'
        plt.title(title)
        plt.axis('off')
    plt.show()
