# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from PIL import Image, ImageDraw
import numpy as np
import pyautogui
import cv2
from collections import deque
import ctypes
import keyboard


# Function to capture and process the screenshot
user32 = ctypes.windll.user32
screen_wc = int(user32.GetSystemMetrics(0) / 2)
screen_hc = int(user32.GetSystemMetrics(1) / 2)
game_w = 1280
game_h = 720
whiteborder_h = 16
screenshot_boundaries = (int(screen_wc - game_w / 2), int(screen_hc - game_h / 2), game_w, game_h - whiteborder_h)

WINDOW_LENGTH = 4  # Number of frames to stack
frame_queue = deque(maxlen=WINDOW_LENGTH)  # Queue to hold the last N frames


def process_and_stack_frames(output_filename_stacks, output_filename_screenshot, input_shape=(80, 45)):
    # Capture the screenshot
    image = pyautogui.screenshot(region=screenshot_boundaries)

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Create a mask for the blue channel with a threshold
    ret, mask = cv2.threshold(image[:, :, 0], 200, 255, cv2.THRESH_BINARY)

    # Prepare a 3-channel mask for bitwise operations
    mask3 = np.zeros_like(image)
    mask3[:, :, 0] = mask  # Apply mask to the blue channel

    # Extract the blue region using bitwise_and
    blue_region = cv2.bitwise_and(image, mask3)

    # Convert the original image to grayscale and then back to BGR
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # Extract the non-blue region from the grayscale image
    non_blue_region = cv2.bitwise_and(gray_image_bgr, 255 - mask3)

    # Combine the blue and grayscale regions
    combined_image = blue_region + non_blue_region

    # Convert to PIL Image to use the resize and convert functions
    image = Image.fromarray(combined_image)

    # Resize and convert to grayscale
    image = image.resize(input_shape)

    # Convert back to NumPy array and save
    processed_image = np.array(image)
    cv2.imwrite(output_filename_screenshot, processed_image)

    processed_image_normalized = np.array(image) / 255.0

    # Add the processed frame to the queue
    frame_queue.append(processed_image_normalized)

    # Stack frames
    if len(frame_queue) == WINDOW_LENGTH:
        stacked_frames = np.stack(frame_queue, axis=-1)
        np.save(output_filename_stacks, stacked_frames)

    return processed_image_normalized


try:
    while True:
        if keyboard.is_pressed('q'):
            print("Aborted")
            break
        for index in range(4):
            # Define a file name for each screenshot
            filename = f"processed_screenshot_{index}.png"

            # Process the screenshot and save
            curr_processed_image = process_and_stack_frames("stacked_frames", filename)

except KeyboardInterrupt:
    print("Program terminated by user.")
