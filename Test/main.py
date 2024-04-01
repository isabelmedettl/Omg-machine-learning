# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
from PIL import Image
import numpy as np
import pyautogui
import cv2
import time
import ctypes
import keyboard  # You might need to install this package using pip
import pygetwindow as gw
import subprocess
import psutil
from collections import deque
import random
import pydirectinput as pdi
import array

user32 = ctypes.windll.user32
screen_wc = int(user32.GetSystemMetrics(0) / 2)
screen_hc = int(user32.GetSystemMetrics(1) / 2)
game_w = 1280
game_h = 720
screenshot_boundaries = (int(screen_wc - game_w / 2), int(screen_hc - game_h / 2), game_w, game_h)

WINDOW_LENGTH = 4  # Number of frames to stack
frame_queue = deque(maxlen=WINDOW_LENGTH)  # Queue to hold the last N frames


# Function to capture and process the screenshot
user32 = ctypes.windll.user32
screen_wc = int(user32.GetSystemMetrics(0)/2)
screen_hc = int(user32.GetSystemMetrics(1)/2)
game_w = 1280
game_h = 720
whiteborder_h = 16
screenshot_boundaries = (int(screen_wc - game_w/2), int(screen_hc - game_h/2), game_w,  game_h - whiteborder_h)

WINDOW_LENGTH = 4  # Number of frames to stack
frame_queue = deque(maxlen=WINDOW_LENGTH)  # Queue to hold the last N frames
cached_distances_to_targets = []


   # Function to capture and process the screenshot
def  process_and_stack_frames(output_filename_stacks, output_filename_screenshot, input_shape=(80, 44)):
    # Capture the screenshot
    image = pyautogui.screenshot(region=screenshot_boundaries)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Convert to PIL Image to use the resize and convert functions
    image = Image.fromarray(image)

    # Resize and convert to grayscale
    image = image.resize(input_shape)

    # Convert back to NumPy array and save
    processed_image = np.array(image)
    cv2.imwrite(output_filename_screenshot, processed_image)

    processed_image_normalized = np.array(image) / 255.0

    pick_up_target_gray = 200.0/255.0
    pick_up_locations = find_pixels_by_color_vectorized(processed_image_normalized, pick_up_target_gray)

    agent_gray = range(80, 120)
    agent_location = find_player_by_color_range(processed_image_normalized, agent_gray)

    if agent_location is not None:
        update_distances_to_targets(pick_up_locations, agent_location[0])


    # Add the processed frame to the queue
    frame_queue.append(processed_image_normalized)

    # Stack frames
    if len(frame_queue) == WINDOW_LENGTH:
        stacked_frames = np.stack(frame_queue, axis=-1)
        np.save(output_filename_stacks, stacked_frames)

# Start the game or application and get the process id


game_path = "C:\\Users\\isabe\\Documents\\ML\\Omg-machine-learning\\c++_mojosabel\\build\\debug\\play.exe"  # Replace this with your game path
os.startfile(game_path)
time.sleep(2)

window_title = "C:\\Users\\isabe\\Documents\\ML\\Omg-machine-learning\\c++_mojosabel\\build\\debug\\play.exe"  # Replace this with the actual title of your game window
other_window_title = "Mojosabel"

print("Starting screenshot capture. The loop will abort when the program stops running.")


# Function to check if a process is still running
def is_process_running(pid):
    return psutil.pid_exists(pid)


actions = [["w"], ["a"], ["s"], ["d"], ["space"], ["w", "a"], ["w", "d"], ["s", "a"], ["s", "d"], ["w", "space"],
           ["a", "space"], ["s", "space"], ["d", "space"], ["w", "a", "space"], ["w", "d", "space"],
           ["s", "a", "space"], ["s", "d",
                                 "space"], ]  # define possible actions, is currently missing diagonal moves and None prob doesn't work as no input


def random_action(delay):
    action = random.choice(actions)

    for x in action:
        pdi.keyDown(x)
    time.sleep(delay)
    for y in action:
        pdi.keyUp(y)


def calculate_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def update_distances_to_targets(pick_up_locations, agent_location):
    print("cached", cached_distances_to_targets)
    distances_to_targets = []
    for loc in pick_up_locations:
        distances_to_targets.append(calculate_distance(loc, agent_location))

    distances_to_targets.sort(reverse=True)
    print("distances",  distances_to_targets)

    #check if any pick-ups are gone
    if len(cached_distances_to_targets) > 0:
        for i in range(len(distances_to_targets)):
            if distances_to_targets[i] < cached_distances_to_targets[i]:
                print("yay rewARD")
                cached_distances_to_targets[i] = distances_to_targets[i]
    else:
        for i in range(len(distances_to_targets)):
            cached_distances_to_targets.append(distances_to_targets[i])


def find_pixels_by_intensity(image_array, target_intensity):
    """
    Find pixels in a grayscale image that match a given intensity value.

    :param image_array: NumPy array of the grayscale image.
    :param target_intensity: The intensity value to find (0 to 255 for 8-bit images).
    :return: List of tuples, each tuple represents the (x, y) position of a matching pixel.
    """
    # Ensure the target intensity is within the valid range for grayscale images
    if not 0 <= target_intensity <= 255:
        raise ValueError("Target intensity must be between 0 and 255.")

    # Find where the image array matches the target intensity
    matches = np.where(image_array == target_intensity)

    # 'matches' is a tuple containing two arrays: one for the y indices and one for the x indices of matching pixels
    # Zip these arrays to get a list of (x, y) positions
    matching_positions = list(zip(matches[1], matches[0]))  # Note the order of indices is reversed (x, y)

    print("machy possy")
    print(matching_positions)
    return matching_positions


def find_player_by_color_range(image_array, player_range):
    for i in player_range:
        position = find_pixels_by_color_vectorized(image_array, i / 255.0)
        if len(position) > 0:
            return position

    print("no pos")
    return None

def find_pixels_by_color_vectorized(image_array, target_gray):
    pos_i_count = 0
    pos_j_count = 0
    positions = []
    for i in image_array:
        for j in i:
            if j == target_gray:
                positions.append((pos_i_count, pos_j_count))
            pos_j_count += 1

        pos_i_count += 1
        pos_j_count = 0

    return positions


# Infinite loop to continuously capture and overwrite screenshots
try:
    while True:
        if not gw.getWindowsWithTitle(window_title) or not gw.getWindowsWithTitle(other_window_title):
            print("Game process closed. Aborting screenshot capture.")
            break

        for index in range(4):
            # Define a file name for each screenshot
            filename = f"processed_screenshot_{index}.png"

            # Process the screenshot and save
            process_and_stack_frames("stacked_frames", filename)

            random_action(0.125)  # Adjust the sleep time as needed


    #print(np.load("stacked_frames.npy"))
except KeyboardInterrupt:
    print("Program terminated by user.")


def is_process_running(pid):
    return psutil.pid_exists(pid)
