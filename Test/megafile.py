import os
from PIL import Image
import numpy as np
import pyautogui
import cv2
import time
import ctypes
import pygetwindow as gw
from collections import deque
import pydirectinput as pdi
import gym
from gym.spaces import Discrete, Box
from gymnasium.envs.registration import register
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Activation
from keras.optimizers import Adam

# Isabel path: C:\\Users\\isabe\\Documents\\ML\\Omg-machine-learning\\c++_mojosabel\\build\\debug\\play.exe
# Monty path: C:\\Python\\GitHub\\Omg-machine-learning\\c++_mojosabel\\build\\debug\\play.exe
game_path = "C:\\Python\\GitHub\\Omg-machine-learning\\c++_mojosabel\\build\\debug\\play.exe"  # Replace this with your game path

# Name of game window
window_title = "Mojosabel"

register(
    id="Mojosabel-v0",
    entry_point="Test.environment_omgml:Environment"  # This path could be wrong
)


class Environment(gym.Env):
    def __init__(self):
        super(Environment, self).__init__()
        self.action_space = Discrete(18)
        self.observation_space = Box(low=0, high=1, shape=(80, 44, 3), dtype=np.float32)  # Stack frames later

        # Initialize the state variables
        self.cached_distances_to_targets = []
        self.distances_to_targets = []  # Is this necessary?
        self.locations = []
        self.pick_up_locations = []
        self.agent_location = (0, 0)
        self.white_pixels = 0
        self.black_pixels = 0
        self.blue_min = 0.854
        self.blue_max = 0.89
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
        if len(self.distances_to_targets) > 0:
            self.distances_to_targets.clear()

        white_pixels_premove = self.white_pixels
        black_pixels_premove = self.black_pixels

        if self.locations[1] is not None:
            for x in action:
                pdi.keyDown(x)
            observation = self.update_locations()  # This might be replaced with an actual stable delay
            for y in action:
                pdi.keyUp(y)
        else:
            observation = self.update_locations()

        for loc in self.pick_up_locations:
            self.distances_to_targets.append(self.calculate_distance(loc, self.agent_location))
        self.distances_to_targets.sort(reverse=True)

        reward = self.calculate_reward(white_pixels_premove, black_pixels_premove)

        if reward >= 100:
            done = True

        info = self.get_info()
        terminated = self.check_goal_state()

        truncated = False
        self.step_counter += 1
        if self.step_counter >= 2000:
            truncated = True

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if gw.getWindowsWithTitle(game_path) or gw.getWindowsWithTitle(window_title):
            os.system(f"taskkill /f /im play.exe")  # Stops the game

        self.step_counter = 0
        self.white_pixels = 0
        self.black_pixels = 0
        self.cached_distances_to_targets.clear()

        os.startfile(game_path)
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

    '''def OLD_stack_frames(self, processed_image_normalized, output_filename_stacks):
        self.frame_queue.append(processed_image_normalized)

        # Stack frames
        if len(self.frame_queue) == self.WINDOW_LENGTH:
            stacked_frames = np.stack(self.frame_queue, axis=-1)
            np.save(output_filename_stacks, stacked_frames)

        return stacked_frames'''

    def update_locations(self):
        # Process the screenshot and save
        curr_processed_image = self.process_screenshot("screenshot00")
        locations = self.find_pixels_by_color_vectorized(curr_processed_image)
        self.pick_up_locations = locations[0]
        self.agent_location = locations[1]

        return curr_processed_image

    def calculate_distance(self, point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    def calculate_reward(self, white_pixels_premove, black_pixels_premove):
        reward = 0

        if len(self.cached_distances_to_targets) > 0:
            for i in range(len(self.distances_to_targets)):
                if self.distances_to_targets[i] < self.cached_distances_to_targets[i]:
                    print("Should get reward")
                    reward += 1
                    self.cached_distances_to_targets[i] = self.distances_to_targets[i]
        else:
            for i in range(len(self.distances_to_targets)):
                self.cached_distances_to_targets.append(self.distances_to_targets[i])

        if self.white_pixels > white_pixels_premove:
            reward += 25  # Reward for progress (minerals / crocodiles)

        if self.black_pixels > black_pixels_premove:
            reward += 100  # Reward for level up

        reward -= 1

        return reward

    def get_info(self):
        return None

    def check_goal_state(self):
        if self.black_pixels >= 2200:
            return True

        return False


env = Environment()
height, width, channels = env.observation_space.shape
actions = env.action_space


def build_model(height, width, channels, action_space):
    model = Sequential()  # According to Tensorflow, sequential is only appropriate when the model has ONE input and ONE output, we have many more. Maybe reconsider.
    model.add(Conv2D(16, (8, 8), strides=(4, 4), input_shape=(18, height, width,
                                                              channels)))  # Because we use images, we need to first set up a convolutional network and then flatten it down. Input_shape is image
    model.add(Activation('relu'))
    model.add(Conv2D(32, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(action_space))  # Action_space is how many actions we have
    model.compile(loss='mse', optimizer=Adam(lr=0.005))
    return model


model = build_model(height, width, channels, actions)

model.summary()


def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2,
                                  nb_steps=1000)
    memory = SequentialMemory(limit=100, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=100
                   )
    return dqn


dqn = build_agent(model, actions)
dqn.compile(Adam(lr=5 * 1e-4))

dqn.fit(env, nb_steps=1000, visualize=False, verbose=2)
dqn.save_weights('SavedWeights/1k-test/dqn_weights.h5')
