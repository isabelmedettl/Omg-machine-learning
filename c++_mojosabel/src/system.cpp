#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <iostream>

#include "System.h"

namespace mojosabel {

    System::System()
    {
        std::cout << "Hej det funkar: systemet!\n";
        SDL_Init(SDL_INIT_EVERYTHING);
        win = SDL_CreateWindow("Mojosabel", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, SCREEN_WIDTH, SCREEN_HEIGHT, 0);
        //SDL_SetWindowFullscreen(win, SDL_WINDOW_FULLSCREEN_DESKTOP);
        ren = SDL_CreateRenderer(win, -1, 0);
        TTF_Init();
        font = TTF_OpenFont((constants::gResPath + "fonts/arial.ttf").c_str(), 36);
        std::cout << "resPath: " << constants::gResPath << std::endl;
        //keyboard[MAX_KEYBOARD_KEYS] = {0};
    }

    System::~System()
    {
        TTF_CloseFont(font);
        TTF_Quit();
        SDL_DestroyWindow(win);
        SDL_DestroyRenderer(ren);
        SDL_Quit();
    }

    SDL_Renderer* System::getRen() const 
    {
        return ren;
    }
    
    TTF_Font* System::getFont() const 
    {
        return font;
    }

    SDL_Window* System::getWin() const 
    {
        return win;
    }

    bool System::isOutOfBounds(int x, int y)
    {
        if (x > SCREEN_WIDTH || x < 0 || y > SCREEN_HEIGHT || y < 0)
        {
            return true;
        }
        return false;
    }
    
    System sys;    
}