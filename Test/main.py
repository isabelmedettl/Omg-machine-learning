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

user32 = ctypes.windll.user32
screen_wc = int(user32.GetSystemMetrics(0)/2)
screen_hc = int(user32.GetSystemMetrics(1)/2)
game_w = 1280
game_h = 720
screenshot_boundaries = (int(screen_wc - game_w/2), int(screen_hc - game_h/2), game_w,  game_h)

WINDOW_LENGTH = 4  # Number of frames to stack
frame_queue = deque(maxlen=WINDOW_LENGTH)  # Queue to hold the last N frames


# Function to capture and process the screenshot
def process_and_stack_frames(output_filename_stacks, output_filename_screenshot, input_shape=(80, 45)):
    # Capture the screenshot
    image = pyautogui.screenshot(region=screenshot_boundaries)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert to PIL Image to use the resize and convert functions
    image = Image.fromarray(image)

    # Resize and convert to grayscale
    image = image.resize(input_shape).convert('L')

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

# Start the game or application and get the process id

game_path = "C:\\Python\\GitHub\\Omg-machine-learning\\c++_mojosabel\\build\\debug\\play.exe"  # Replace this with your game path
os.startfile(game_path)
time.sleep(1)

window_title = "C:\\Python\\GitHub\\Omg-machine-learning\\c++_mojosabel\\build\debug\\play.exe"  # Replace this with the actual title of your game window
other_window_title = "Mojosabel"

print("Starting screenshot capture. The loop will abort when the program stops running.")

# Function to check if a process is still running
def is_process_running(pid):
    return psutil.pid_exists(pid)

actions = [["w"], ["a"], ["s"], ["d"], ["space"], ["w", "a"], ["w", "d"], ["s", "a"], ["s", "d"], ["w", "space"], ["a", "space"], ["s", "space"], ["d", "space"], ["w", "a", "space"], ["w", "d", "space"], ["s", "a", "space"], ["s", "d", "space"],]  # define possible actions, is currently missing diagonal moves and None prob doesn't work as no input

def random_action(delay):
    action = random.choice(actions)

    for x in action:
        pdi.keyDown(x)
    time.sleep(delay)
    for y in action:
        pdi.keyUp(y)


# Infinite loop to continuously capture and overwrite screenshots

try:
    while True:
        if not gw.getWindowsWithTitle(window_title) or not gw.getWindowsWithTitle(other_window_title):
            print("Game process closed. Aborting screenshot capture.")
            break

        for i in range(4):
            # Define a file name for each screenshot
            filename = f"processed_screenshot_{i}.png"

            # Process the screenshot and save
            process_and_stack_frames("stacked_frames", filename)

            random_action(0.125)  # Adjust the sleep time as needed

    print(np.load("stacked_frames.npy"))
except KeyboardInterrupt:
    print("Program terminated by user.")

def is_process_running(pid):
    return psutil.pid_exists(pid)

