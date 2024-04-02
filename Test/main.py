# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
from PIL import Image, ImageDraw
import numpy as np
import pyautogui
import cv2
import time
import ctypes
import pygetwindow as gw
import psutil
from collections import deque
import random
import pydirectinput as pdi
import matplotlib.pyplot as plt
import keyboard

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
distances_to_targets = []


pick_up_locations = []
ds = (0, 0)

# gray constants
agent_gray = range(80, 120)
pick_up_target_gray = 200.0 / 255.0


score = 0


#range
blue_min = 0.854
blue_max = 0.89
agent_min = 0.31
agent_max = 0.471


def  process_and_stack_frames(output_filename_stacks, output_filename_screenshot, input_shape=(80, 44)):
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

    return action


def calculate_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def debug_visualization_image(processed_image, it_counter):
    """
    Saves the debug visualization as an image.

    :param agent_location: The detected location of the agent in the image.
    :param pick_up_locations: List of detected pickup locations in the image.
    :param processed_image: The processed image (numpy array).
    :param distances_to_targets: Calculated distances to targets for annotation.
    :param iteration: Current iteration or frame number to uniquely name the output file.
    """
    # Convert processed image to RGB for plotting
    if len(processed_image.shape) == 2:  # Grayscale to RGB
        processed_image = np.stack((processed_image,) * 3, axis=-1)

    fig, ax = plt.subplots()
    ax.imshow(processed_image, cmap='gray')

    # Plot agent location
    if agent_location:
        ax.scatter(agent_location[0][1], agent_location[0][0], color='red', s=100, label='Agent', edgecolors='w')

    # Plot pickup locations
    for loc in pick_up_locations:
        ax.scatter(loc[1], loc[0], color='lime', s=100, label='Pickup', edgecolors='w')
        # Draw line from agent to pickup location
        if agent_location:
            ax.plot([agent_location[0][1], loc[1]], [agent_location[0][0], loc[0]], color='yellow', linestyle='-',
                    linewidth=1)

    # Remove axis
    plt.axis('off')

    # Save plot to a PNG file
    plt.savefig(f"debug_frame_{iteration_counter}.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Reopen the saved image to annotate with PIL
    image = Image.open(f"debug_frame_{iteration_counter}.png")
    draw = ImageDraw.Draw(image)

    # Prepare text to overlay
    text = f"Distances: {distances_to_targets}\nCached: {cached_distances_to_targets}"

    # Add text overlay to the image in the top-left corner
    draw.text((10, 10), text, fill=(255, 255, 255))

    # Save the annotated image
    image.save(f"annotated_debug_frame_{iteration_counter}.png")


def observe_step(delay):

    reward = 0
    print("cached", cached_distances_to_targets)
    distances_to_targets.clear()

    for loc in pick_up_locations:
        distances_to_targets.append(calculate_distance(loc, agent_location))

    distances_to_targets.sort(reverse=True)
    print("distances",  distances_to_targets)

    action = random_action(delay)

    #check if any pick-ups are gone
    if len(cached_distances_to_targets) > 0:
        for i in range(len(distances_to_targets)):
            if len(distances_to_targets) > len(cached_distances_to_targets):
                print("cached longer than distances", distances_to_targets[i], "dist", len(distances_to_targets), "cached", len(cached_distances_to_targets))
            elif distances_to_targets[i] < cached_distances_to_targets[i]:
                reward += 1
                cached_distances_to_targets[i] = distances_to_targets[i]
    else:
        for i in range(len(distances_to_targets)):
            cached_distances_to_targets.append(distances_to_targets[i])

    return reward


def find_player_by_color_range(image_array, player_range):
    for i in player_range:
        position = find_pixels_by_color_vectorized(image_array, i / 255.0)
        if len(position) > 0:
            return position

    print("no pos")
    return None


def find_pixels_by_color_vectorized(image_array):
    pos_i_count = 0
    pos_j_count = 0
    positions = []
    agent_position = None
    for i in image_array:
        for j in i:
            if blue_min < j[0] < blue_max: #blue
                positions.append((pos_i_count, pos_j_count))
            elif agent_min < j[1] < agent_max: #grey
                agent_position = (pos_i_count, pos_j_count)

            pos_j_count += 1
        pos_i_count += 1
        pos_j_count = 0

    return [positions, agent_position]


# Infinite loop to continuously capture and overwrite screenshots
try:
    while True:
        if not gw.getWindowsWithTitle(window_title) or not gw.getWindowsWithTitle(other_window_title):
            print("Game process closed. Aborting screenshot capture.")
            print("Score:", score)
            break

        iteration_counter = 0
        for index in range(4):
            # Define a file name for each screenshot
            filename = f"processed_screenshot_{index}.png"

            # Process the screenshot and save
            curr_processed_image = process_and_stack_frames("stacked_frames", filename)

            locations = find_pixels_by_color_vectorized(curr_processed_image)
            pick_up_locations = locations[0]
            agent_location = locations[1]

            # debug_visualization_image(curr_processed_image, iteration_counter)

            if agent_location is not None:
                score += observe_step(0.125)

            # Debug visualization call
            iteration_counter += 1
            #random_action(0.125)  # Adjust the sleep time as needed


    #print(np.load("stacked_frames.npy"))
except KeyboardInterrupt:
    print("Program terminated by user.")


def is_process_running(pid):
    return psutil.pid_exists(pid)
