import tensorflow as tf
import matplotlib.pyplot as plt

# Get the data and split it into training and testing sets
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Scale down the values for easy processing
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape the images to have a single channel (grayscale) and add a new dimension
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Visualize the data
print(train_images.shape)
print(test_images.shape)
print(train_labels)

# Define the CNN model
cnn_model = tf.keras.models.Sequential()
cnn_model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
cnn_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
cnn_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
cnn_model.add(tf.keras.layers.Flatten())
cnn_model.add(tf.keras.layers.Dense(128, activation='relu'))
cnn_model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the CNN model
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
cnn_model.fit(train_images, train_labels, epochs=3)

# Test the CNN model
val_loss, val_acc = cnn_model.evaluate(test_images, test_labels)
print('Test accuracy: ', val_acc)

# Save the CNN model
cnn_model.save('mnist_cnn_model')
