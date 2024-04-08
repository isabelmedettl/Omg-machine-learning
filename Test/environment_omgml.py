import os
from PIL import Image, ImageDraw
import numpy as np
import pyautogui
import cv2
import time
import ctypes
import pygetwindow as gw
import subprocess
from collections import deque
import pydirectinput as pdi
import gymnasium
from gymnasium.spaces import Discrete, Box
from gymnasium.envs.registration import register
import mss.tools

pdi.PAUSE = 0.0001

# Isabel path: C:\\Users\\isabe\\Documents\\ML\\Omg-machine-learning\\c++_mojosabel\\build\\debug\\play.exe
# Skolsabel path: C:\\Users\\mijo1919\\Documents\\Omg-machine-learning\\c++_mojosabel\\build\\debug\\play.exe
# Monty path: C:\\Python\\GitHub\\Omg-machine-learning\\c++_mojosabel\\build\\debug\\play.exe
game_path = "C:\\Python\\GitHub\\Omg-machine-learning\\c++_mojosabel\\build\\debug\\play.exe"  # Replace this with your game path

# Name of game window
window_title = "Mojosabel"

fps = 120

register(
    id="Mojosabel-v0",
    entry_point="Test.environment_omgml:Environment"  # This path could be wrong
)


class Environment(gymnasium.Env):
    def __init__(self):
        super(Environment, self).__init__()
        self.action_space = Discrete(18)
        self.observation_space = Box(low=0, high=1, shape=(44, 80, 3), dtype=np.float32)  # Stack frames later


        # Initialize the state variables
        self.cached_distances_to_targets = []
        self.distances_to_targets = []  # Is this necessary?
        self.locations = []
        self.pick_up_locations = []
        self.agent_location = (0, 0)
        self.white_pixels = 0
        self.black_pixels = 0
        self.blue_min = 0.721
        self.blue_max = 0.73
        self.agent_min = 0.28
        self.agent_max = 0.51
        self.step_counter = 0

        self.WINDOW_LENGTH = 4  # Number of frames to stack
        self.frame_queue = deque(maxlen=self.WINDOW_LENGTH)  # Queue to hold the last N frames

        # Screenshot variables
        self.user32 = ctypes.windll.user32
        self.screen_wc = int(self.user32.GetSystemMetrics(0) / 2)
        self.screen_hc = int(self.user32.GetSystemMetrics(1) / 2)
        self.game_w = 1280
        self.game_h = 720
        self.whiteborder_h = 16
        self.screenshot_boundaries = {"left":int(self.screen_wc - self.game_w / 2), "top":int(self.screen_hc - self.game_h / 2), "width":self.game_w, "height":self.game_h - self.whiteborder_h}


        # Define the actions as a class attribute
        self.actions = [["w"], ["a"], ["s"], ["d"], ["space"], ["w", "a"], ["w", "d"], ["s", "a"], ["s", "d"],
                        ["w", "space"],
                        ["a", "space"], ["s", "space"], ["d", "space"], ["w", "a", "space"], ["w", "d", "space"],
                        ["s", "a", "space"], ["s", "d", "space"], [None]]

        self.game_process = None


    def step(self, action_index):
        global fps

        #print("took step", self.actions[action_index])
        action = self.actions[action_index]
        if len(self.distances_to_targets) > 0:
            self.distances_to_targets.clear()

        white_pixels_premove = self.white_pixels
        black_pixels_premove = self.black_pixels

        #print("locs", self.locations)

        prepretime = time.time()

        input_frames = 12

        if not len(self.locations) <= 1:
            for x in self.actions[action_index]:
                pdi.keyDown(x)
            pretime = time.time()
            observation = self.update_locations()  # This might be replaced with an actual stable delay
            time.sleep(max(0.0, input_frames/fps-(time.time() - pretime)))
            for y in self.actions[action_index]:
                pdi.keyUp(y)
        else:
            observation = self.update_locations()

        for loc in self.pick_up_locations:
            self.distances_to_targets.append(self.calculate_distance(loc, self.agent_location))
        self.distances_to_targets.sort(reverse=True)


        reward = self.calculate_reward(white_pixels_premove, black_pixels_premove)

        if reward >= 50:
            done = True

        info = self.get_info()
        terminated = self.check_goal_state()

        truncated = False
        self.step_counter += 1
        if self.step_counter >= 2000:
            truncated = True

        print("Minerals: ", self.locations[0], "Agent: ", self.locations[1])

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        print("it reset")
        #if gw.getWindowsWithTitle(game_path) or gw.getWindowsWithTitle(window_title):
          #  os.system(f"taskkill /f /im play.exe")  # Stops the game

        # If there's an existing game process, terminate it
        if self.game_process is not None:
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.game_process.pid)])
            #self.game_process.terminate()  # Gracefully terminate the process
            #self.game_process.wait()  # Wait for the game process to terminate
            self.game_process = None  # Reset the game process variable


        self.step_counter = 0
        self.white_pixels = 0
        self.black_pixels = 0
        self.cached_distances_to_targets.clear()

       # os.startfile(game_path)
        # Start a new game process
        self.game_process = subprocess.Popen([game_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)

        observation = self.update_locations()
        info = self.get_info()

        return observation, info

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

    def process_screenshot(self, output_filename_screenshot, input_shape=(80, 44)):
        # Capture the screenshot
        with mss.mss() as sct:
            image = sct.grab(self.screenshot_boundaries)

        # Convert to NumPy array and to BGR color space
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGRA2BGR)

        # Resize early to reduce the amount of data processed in subsequent steps
        # image = cv2.resize(image, input_shape)

        image = Image.fromarray(image)

        # Resize and convert to grayscale
        image = image.resize(input_shape)

        # Create a mask for the blue channel with a threshold
        # ret, mask = cv2.threshold(image[:, :, 0], 200, 255, cv2.THRESH_BINARY)

        # Prepare a 3-channel mask for bitwise operations
        # mask3 = np.zeros_like(image)
        # mask3[:, :, 0] = mask  # Apply mask to the blue channel

        # Extract the blue region using bitwise_and
        # blue_region = cv2.bitwise_and(image, mask3)

        # Convert the original image to grayscale and then back to BGR
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray_image_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        # Extract the non-blue region from the grayscale image
        # non_blue_region = cv2.bitwise_and(gray_image_bgr, 255 - mask3)

        # Combine the blue and grayscale regions
        # combined_image = blue_region + non_blue_region

        # Convert back to NumPy array and save
        processed_image = np.array(image)
        # cv2.imwrite(output_filename_screenshot, processed_image)

        # Normalize the processed image
        processed_image_normalized = processed_image / 255.0

        return processed_image_normalized

    '''def OLD_stack_frames(self, processed_image_normalized, output_filename_stacks):
        self.frame_queue.append(processed_image_normalized)

        # Stack frames
        if len(self.frame_queue) == self.WINDOW_LENGTH:
            stacked_frames = np.stack(self.frame_queue, axis=-1)
            np.save(output_filename_stacks, stacked_frames)

        return stacked_frames'''

    '''    def process_screenshot(self, output_filename_screenshot, input_shape=(80, 44)):
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

        return processed_image_normalized'''

    def update_locations(self):
        # Process the screenshot and save
        curr_processed_image = self.process_screenshot("screenshot00.png")
        self.locations = self.find_pixels_by_color_vectorized(curr_processed_image)
        self.pick_up_locations = self.locations[0]
        self.agent_location = self.locations[1]

        return curr_processed_image

    def calculate_distance(self, point1, point2):
        if point2 is None or len(point2) == 0 or point2[0] is None:
            return 100  # Returns big distance to make sure we no say good when no good
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    def calculate_reward(self, white_pixels_premove, black_pixels_premove):

        # print(self.cached_distances_to_targets, self.distances_to_targets)

        reward = 0

        if len(self.cached_distances_to_targets) > 0:
            for i in range(min(len(self.distances_to_targets), len(self.cached_distances_to_targets))):
                if self.distances_to_targets[i] < self.cached_distances_to_targets[i]:
                    reward += (self.cached_distances_to_targets[i] - self.distances_to_targets[i]) / 5 # we are testing how much to divide
                    self.cached_distances_to_targets[i] = self.distances_to_targets[i]
        else:
            for i in range(len(self.distances_to_targets)):
                self.cached_distances_to_targets.append(self.distances_to_targets[i])

        if self.white_pixels > white_pixels_premove:
            reward += 10  # Reward for progress (minerals / crocodiles)

        if self.black_pixels > 400:
            reward -= 100  # Reward for death (punishment)
            print("Reward is -100 cus dead")
        elif self.black_pixels > black_pixels_premove:
            reward += 50  # Reward for level up

        reward -= 0.1

        return reward

    def get_info(self):
        return None

    def check_goal_state(self):
        if self.black_pixels >= 2200:
            return True

        return False


