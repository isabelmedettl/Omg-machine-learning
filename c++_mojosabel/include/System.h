#ifndef SYSTEM_H
#define SYSTEM_H

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include "Constants.h"

namespace mojosabel
{
    class System 
    {
    private:
        SDL_Window* win;
        SDL_Renderer* ren;
        TTF_Font* font;
        int bSaveFolderExists = 0;

    public:
        System();
        System(const System* other) = delete;
        const System operator=(const System& rhs) = delete;
        ~System();
        SDL_Renderer* getRen() const;
        TTF_Font* getFont() const;
        SDL_Window* getWin() const;
        int keyboard[MAX_KEYBOARD_KEYS] = {0};
        bool isOutOfBounds(int x, int y);
        int saveFolderExists();
        void cleanDirectory();

    private:
        void ensureFolderExists();
        int saveFolderExists_internal();
        void cleanDirectory_internal();
    };

    extern System sys;

} 

#endif