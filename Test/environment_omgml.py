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
import gym
from gym.spaces import Discrete, Box


class Environment(gym.Env):
    def __init__(self):
        super(Environment, self).__init__()
        self.action_space = Discrete(18)
        self.observation_space = Box(low=0, high=1, shape=(80, 44, 3), dtype=np.float32)  # Stack frames later

        # Initialize the state variables
        self.cached_distances_to_targets = []
        self.locations = []
        self.pick_up_locations = []
        self.agent_location = (0, 0)
        self.white_pixels = 0
        self.black_pixels = 0
        self.blue_min = 0.854
        self.blue_max = 0.89
        self.agent_min = 0.28
        self.agent_max = 0.51

        self.WINDOW_LENGTH = 4  # Number of frames to stack
        self.frame_queue = deque(maxlen=self.WINDOW_LENGTH)  # Queue to hold the last N frames

        # Screenshot variables
        self.user32 = ctypes.windll.user32
        self.screen_wc = int(self.user32.GetSystemMetrics(0) / 2)
        self.screen_hc = int(self.user32.GetSystemMetrics(1) / 2)
        self.game_w = 1280
        self.game_h = 720
        self.whiteborder_h = 16
        self.screenshot_boundaries = (
        int(self.screen_wc - self.game_w / 2), int(self.screen_hc - self.game_h / 2), self.game_w,
        self.game_h - self.whiteborder_h)

        # Define the actions as a class attribute
        self.actions = [["w"], ["a"], ["s"], ["d"], ["space"], ["w", "a"], ["w", "d"], ["s", "a"], ["s", "d"],
                        ["w", "space"],
                        ["a", "space"], ["s", "space"], ["d", "space"], ["w", "a", "space"], ["w", "d", "space"],
                        ["s", "a", "space"], ["s", "d", "space"], [None]]

    def step(self, action_index):
        action = self.actions[action_index]
        reward = 0
        distances_to_targets.clear()



        return observation, reward, done

    def render(self):
        pass

    def reset(self):
        pass
        # kill exe
        # reset global valutes
        # start exe
        # start stepping

    def find_pixels_by_color_vectorized(self, image_array):
        pos_i_count = 0
        pos_j_count = 0
        positions = []
        curr_white_pixels = 0
        curr_black_pixels = 0
        agent_position = None

        for i in image_array:
            for j in i:
                if self.blue_min < j[0] < self.blue_max:  # blue
                    positions.append((pos_i_count, pos_j_count))
                elif self.agent_min < j[1] < self.agent_max:  # grey
                    agent_position = (pos_i_count, pos_j_count)
                elif j[1] == 1:
                    curr_white_pixels += 1
                elif j[1] == 0:
                    curr_black_pixels += 1

                pos_j_count += 1
            pos_i_count += 1
            pos_j_count = 0

        if curr_white_pixels > self.white_pixels:
            self.white_pixels = curr_white_pixels

        if curr_black_pixels > self.black_pixels:
            self.black_pixels = curr_black_pixels

        return [positions, agent_position]

    def process_and_stack_frames(self, output_filename_screenshot, input_shape=(80, 44)):
        # Capture the screenshot
        image = pyautogui.screenshot(region=self.screenshot_boundaries)

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

        return processed_image_normalized

    def stack_frames(self, processed_image_normalized, output_filename_stacks):
        self.frame_queue.append(processed_image_normalized)

        # Stack frames
        if len(self.frame_queue) == self.WINDOW_LENGTH:
            stacked_frames = np.stack(self.frame_queue, axis=-1)
            np.save(output_filename_stacks, stacked_frames)

        return stacked_frames
