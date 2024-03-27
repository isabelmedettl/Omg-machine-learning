# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
from PIL import ImageGrab as IG
import numpy as np
import pyautogui
import cv2
import time
import ctypes

user32 = ctypes.windll.user32
screen_wc = int(user32.GetSystemMetrics(0)/2)
screen_hc = int(user32.GetSystemMetrics(1)/2)
game_w = 1280
game_h = 720
screenshot_boundaries = (int(screen_wc - game_w/2), int(screen_hc - game_h/2), game_w,  game_h)



# def capture_screen(filename):




os.startfile("C:\\Users\\mijo1919\\Documents\\Omg-machine-learning\\c++_mojosabel\\build\debug\\play.exe")

time.sleep(1)

# Capture (left, top, right, bottom)

image = pyautogui.screenshot(region=screenshot_boundaries)
image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
cv2.imwrite("screeshot.png", image)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
