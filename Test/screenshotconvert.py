import numpy as np
from PIL import Image


def load_array_and_save_as_png(array_path, output_image_path):
    # Step 1: Load the NumPy array (assuming it's a .npy file)
    array = np.load(array_path)

    # Step 2: Process the array if necessary (e.g., rescaling)
    # This step is dependent on your specific array and may not always be required.
    # For example, if your array is normalized between 0 and 1, you might want to rescale it to 0-255:
    array = np.clip(array * 255, 0, 255).astype(np.uint8)

    # Step 3: Convert the NumPy array to a PIL Image
    image = Image.fromarray(array)

    # Step 4: Save the PIL Image as a PNG
    image.save(output_image_path, 'PNG')


# Example usage
array_path = 'stacked_frames.npy'  # Update this to the path of your .npy file
output_image_path = 'output_image.png'
# Uncomment the line below to run the function with your file paths
load_array_and_save_as_png(array_path, output_image_path)
