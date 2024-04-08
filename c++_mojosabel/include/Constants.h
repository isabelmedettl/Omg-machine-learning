#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>


namespace constants
{
    #define SCREEN_WIDTH 1280
    #define SCREEN_HEIGHT 720
    #define MAX_KEYBOARD_KEYS 350
    #define TILE_SIZE 32
    #define MAP_WIDTH 40
    #define MAP_HEIGHT 22

    #define PROGRESSION_INCREMENT 3
    #define SLIDER_WIDTH 32
    #define MAX_LEVELS 10

    #define Y_MIN_OFFSET 0 
    #define Y_MAX_OFFSET 22
    #define X_MIN_OFFSET 1 
    #define X_MAX_OFFSET 39

    const std::string playerTag = "Player";
    const std::string shellBulletTag = "Bullet";
    const std::string enemyTag = "Enemy";


    //gResPath-contains the relative path to your resources.
	const std::string gResPath = "resources/";

    //saving bpm and images
	const std::string gSaveImagePath = "saved_images/";
    const std::string saveFileName = "screenshot_";

    //keyboard keys enums
    #define KEY_A 4
    #define KEY_B 5
    #define KEY_C 6
    #define KEY_D 7
    #define KEY_E 8
    #define KEY_F 9
    #define KEY_G 10
    #define KEY_H 11
    #define KEY_I 12
    #define KEY_J 13
    #define KEY_K 14
    #define KEY_L 15
    #define KEY_M 16
    #define KEY_N 17
    #define KEY_O 18
    #define KEY_P 19
    #define KEY_Q 20
    #define KEY_R 21
    #define KEY_S 22
    #define KEY_T 23
    #define KEY_U 24
    #define KEY_V 25
    #define KEY_W 26
    #define KEY_X 27
    #define KEY_Y 28
    #define KEY_Z 29
    #define KEY_1 30
    #define KEY_2 31
    #define KEY_3 32
    #define KEY_4 33
    #define KEY_5 34
    #define KEY_6 35
    #define KEY_7 36
    #define KEY_8 37
    #define KEY_9 38
    #define KEY_0 39
    #define KEY_RETURN 40
    #define KEY_ESC 41
    #define KEY_BACKSPACE 42
    #define KEY_TAB 43
    #define KEY_SPACE 44
    #define KEY_CAPS 57
    #define KEY_RIGHTARROW 79
    #define KEY_LEFTARROW 80
    #define KEY_DOWNARROW 81
    #define KEY_UPARROW 82
    #define KEY_LEFTCTRL 224
    #define KEY_LEFTSHIFT 225
    #define KEY_LEFTALT 226
    #define KEY_RIGHTCTRL 228
    #define KEY_RIGHTSHIFT 229
    #define KEY_RIGHTALT 230
}

#endif