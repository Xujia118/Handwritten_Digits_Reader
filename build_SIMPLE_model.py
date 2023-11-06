import tensorflow as tf
import matplotlib.pyplot as plt

# Get the data and split it into training and testing sets
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Scale down the values for easy processing
train_images = train_images / 255.0
test_images = test_images / 255.0

# Visualize the data
print(train_images.shape)
print(test_images.shape)
print(train_labels)

# Display the first image
plt.imshow(train_images[0], cmap='gray')
plt.show()

# Define the neural network
my_model = tf.keras.models.Sequential()
my_model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
my_model.add(tf.keras.layers.Dense(128, activation='relu'))
my_model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
my_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
my_model.fit(train_images, train_labels, epochs=3)

# Test the model
val_loss, val_acc = my_model.evaluate(test_images, test_labels)
print('Test accurary: ', val_acc)

# Save the model
my_model.save('new_mnist_model')

