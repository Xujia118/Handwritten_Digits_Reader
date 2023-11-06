import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

my_new_model = tf.keras.models.load_model('mnist_cnn_model')

class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

test_folder = 'Test/'
png_file = 'test6.png'
image_path = os.path.join(test_folder, png_file)
img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
img = img / 255.0
img = 1 - img

plt.imshow(img, cmap='gray')
plt.show()

img = np.expand_dims(img, 0)
prediction = my_new_model.predict(img)
index = np.argmax(prediction[0])
result_digit = class_labels[index]
print(result_digit)


