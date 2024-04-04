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

delay_between_actions = 0.5

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

cached_distances_to_targets = []
distances_to_targets = []

pick_up_locations = []
agent_location = (0, 0)
locations = []

# gray constants
agent_gray = range(80, 120)
pick_up_target_gray = 200.0 / 255.0
white_pixels = 0
black_pixels = 0

score = 0

# range
blue_min = 0.854
blue_max = 0.89
agent_min = 0.28
agent_max = 0.51

# Name of game window
window_title = "C:\\Python\\GitHub\\Omg-machine-learning\\c++_mojosabel\\build\\debug\\play.exe"  # Replace this with the actual title of your game window
other_window_title = "Mojosabel"


def process_and_stack_frames(output_filename_stacks, output_filename_screenshot, input_shape=(80, 44)):
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


# Function to check if a process is still running
def is_process_running(pid):
    return psutil.pid_exists(pid)


actions = [["w"], ["a"], ["s"], ["d"], ["space"], ["w", "a"], ["w", "d"], ["s", "a"], ["s", "d"], ["w", "space"],
           ["a", "space"], ["s", "space"], ["d", "space"], ["w", "a", "space"], ["w", "d", "space"],
           ["s", "a", "space"], ["s", "d", "space"], [None]]  # define possible actions


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
    plt.savefig(f"debug_frame_{it_counter}.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Reopen the saved image to annotate with PIL
    image = Image.open(f"debug_frame_{it_counter}.png")
    draw = ImageDraw.Draw(image)

    # Prepare text to overlay
    text = f"Distances: {distances_to_targets}\nCached: {cached_distances_to_targets}"

    # Add text overlay to the image in the top-left corner
    draw.text((10, 10), text, fill=(255, 255, 255))

    # Save the annotated image
    image.save(f"annotated_debug_frame_{it_counter}.png")


def play_step(action, screenshot_index):
    global cached_distances_to_targets
    global pick_up_locations
    global locations
    global agent_location
    reward = 0
    print("cached", cached_distances_to_targets)
    distances_to_targets.clear()

    global white_pixels, black_pixels
    white_pixels_premove = white_pixels
    black_pixels_premove = black_pixels

    print(agent_location)
    if locations[1] is not None:

        for x in action:
            pdi.keyDown(x)
        # time.sleep(delay_between_actions)
        # Define a file name for each screenshot
        filename = f"processed_screenshot_{screenshot_index}.png"

        # Process the screenshot and save
        curr_processed_image = process_and_stack_frames("stacked_frames", filename)

        locations = find_pixels_by_color_vectorized(curr_processed_image)
        pick_up_locations = locations[0]
        agent_location = locations[1]

        for y in action:
            pdi.keyUp(y)

        for loc in pick_up_locations:
            distances_to_targets.append(calculate_distance(loc, agent_location))

        distances_to_targets.sort(reverse=True)
        print("distances", distances_to_targets)

    else:
        print("Player not found, doing random move")
        random_action(0.125)
        filename = f"processed_screenshot_{screenshot_index}.png"

        # Process the screenshot and save
        curr_processed_image = process_and_stack_frames("stacked_frames", filename)
        locations = find_pixels_by_color_vectorized(curr_processed_image)
        pick_up_locations = locations[0]
        agent_location = locations[1]

    # check if any pick-ups are gone
    if len(cached_distances_to_targets) > 0:
        for i in range(len(distances_to_targets)):
            if len(distances_to_targets) > len(cached_distances_to_targets):
                print("cached longer than distances", distances_to_targets[i], "dist", len(distances_to_targets),
                      "cached", len(cached_distances_to_targets))
            elif distances_to_targets[i] < cached_distances_to_targets[i]:
                print("Should get reward")
                reward += 1
                cached_distances_to_targets[i] = distances_to_targets[i]
    else:
        for i in range(len(distances_to_targets)):
            cached_distances_to_targets.append(distances_to_targets[i])

    if white_pixels > white_pixels_premove:
        reward += 25

    if black_pixels > black_pixels_premove:
        reward += 100

    reward -= 1

    return reward


def find_pixels_by_color_vectorized(image_array):
    pos_i_count = 0
    pos_j_count = 0
    positions = []
    curr_white_pixels = 0
    curr_black_pixels = 0
    global white_pixels  # Is this how you do global variables? If something is weird with rewards, check this
    global black_pixels
    agent_position = None

    for i in image_array:
        for j in i:
            if blue_min < j[0] < blue_max:  # blue
                positions.append((pos_i_count, pos_j_count))
            elif agent_min < j[1] < agent_max:  # grey
                agent_position = (pos_i_count, pos_j_count)
            elif j[1] == 1:
                curr_white_pixels += 1
            elif j[1] == 0:
                curr_black_pixels += 1

            pos_j_count += 1
        pos_i_count += 1
        pos_j_count = 0

    if curr_white_pixels > white_pixels:
        white_pixels = curr_white_pixels

    if curr_black_pixels > black_pixels:
        black_pixels = curr_black_pixels

    return [positions, agent_position]


def get_action(state, model):
    state_tensor = np.expand_dims(state, axis=0)  # Prepare state
    q_values = model.predict(state_tensor)  # Get Q-values for each action
    action = np.argmax(q_values[0])  # Select the action with the highest Q-value
    return action


# Infinite loop to continuously capture and overwrite screenshots
def train(episodes):
    global score
    global pick_up_locations, agent_location
    global locations
    global white_pixels, black_pixels
    global cached_distances_to_targets
    results = ""

    for episode in range(1, episodes + 1):
        done = False
        step_counter = 0
        score = 0
        white_pixels = 0
        black_pixels = 0
        cached_distances_to_targets.clear()
        # Start the game or application and get the process id
        # Remember to ALSO CHANGE PATH FOR THE WINDOW AT LINE 55(ish)
        # Isabel path: C:\\Users\\isabe\\Documents\\ML\\Omg-machine-learning\\c++_mojosabel\\build\\debug\\play.exe
        # Monty path: C:\\Python\\GitHub\\Omg-machine-learning\\c++_mojosabel\\build\\debug\\play.exe
        game_path = "C:\\Python\\GitHub\\Omg-machine-learning\\c++_mojosabel\\build\\debug\\play.exe"  # Replace this with your game path
        os.startfile(game_path)
        time.sleep(3)

        filename = f"processed_screenshot_0.png"
        curr_processed_image = process_and_stack_frames("stacked_frames", filename)
        locations = find_pixels_by_color_vectorized(curr_processed_image)
        agent_location = locations[1]

        try:
            while True:
                if not gw.getWindowsWithTitle(window_title) or not gw.getWindowsWithTitle(other_window_title) or done:
                    print("Game process closed. Aborting screenshot capture.")
                    results += f" \nEpisode {episode} - Score: {score}"
                    break

                iteration_counter = 0
                for index in range(4):
                    preloop = time.time()
                    score += play_step(random.choice(actions), index)
                    step_counter += 1
                    print(time.time() - preloop)
                    # debug_visualization_image(curr_processed_image, iteration_counter)

                    # Debug visualization call
                    iteration_counter += 1

                    if step_counter >= 12:
                        done = True
                        print("Should exit game")
                        break

            # print(np.load("stacked_frames.npy"))
        except KeyboardInterrupt:
            print("Program terminated by user.")

        os.system(f"taskkill /f /im play.exe")
    print(results)


def __main__():
    train(3)


if __name__ == "__main__":
    __main__()