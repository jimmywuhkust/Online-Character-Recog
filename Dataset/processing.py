from keras.layers import Flatten
from keras.models import Sequential
import numpy as np
from PIL import Image
import os

# Create a Sequential model with a Flatten layer
model = Sequential()
model.add(Flatten(input_shape=(50, 50)))  # Adjust input_shape to match your image dimensions

# Define the number of images
num_images = 60

# Create a directory to save flattened images
if not os.path.exists('flattened_images'):
    os.makedirs('flattened_images')

# Loop through images and flatten them
for i in range(1, num_images + 1):
    image_path = f'you_dataset-{i:02d}.png'  # Assuming your image files are named like "your_dataset-01.png", "your_dataset-02.png", etc.
    image = Image.open(image_path)
    image_array = np.array(image)

    flattened_image = model.predict(image_array.reshape(1, *image_array.shape))
    flattened_image = flattened_image.reshape(1, -1)

    # Save the flattened image to the 'flattened_images' directory
    flattened_image_path = os.path.join('flattened_images', f'flattened_your_dataset-{i:02d}.png')
    Image.fromarray(flattened_image[0].astype('uint8')).save(flattened_image_path)

# Combine the flattened images into a single file
flattened_image_paths = [os.path.join('flattened_images', f'flattened_your_dataset-{i:02d}.png') for i in range(1, num_images + 1)]

images = [Image.open(path) for path in flattened_image_paths]
widths, heights = zip(*(i.size for i in images))
combined_image = Image.new('RGB', (sum(widths), max(heights)))

x_offset = 0
for image in images:
    combined_image.paste(image, (x_offset, 0))
    x_offset += image.width

combined_image.save('combined_flattened_images.png')
