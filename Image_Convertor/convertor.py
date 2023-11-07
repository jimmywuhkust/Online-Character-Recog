from PIL import Image
import numpy as np

# Load the original data as a 1D array
original_data = np.array(Image.open("character_data_8bytes.png"))

# Define the dimensions of the individual images
individual_image_width = 50
individual_image_height = 50

# Create a directory to save the individual images
import os

if not os.path.exists("output_images"):
    os.makedirs("output_images")

# Calculate the number of individual images
num_images = len(original_data) // (individual_image_width * individual_image_height)

# Loop through the data and split it into individual images
for i in range(num_images):
    # Extract each portion of data for an individual image
    start = i * individual_image_width * individual_image_height
    end = (i + 1) * individual_image_width * individual_image_height
    image_data = original_data[start:end]
    
    # Reshape the data into a 50x50 image
    individual_image = Image.fromarray(image_data.reshape(individual_image_height, individual_image_width))
    
    # Save each individual image
    individual_image.save(f"output_images/output_image_{i}.png")

print("Images have been successfully split and saved in the 'output_images' directory.")
