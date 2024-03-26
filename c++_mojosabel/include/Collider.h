#ifndef COLLIDER_H
#define COLLIDER_H

#include <SDL2/SDL.h>
namespace mojosabel {

    struct Collider 
    {
        int xOffset, yOffset;
        SDL_Rect rect;
        Collider() = delete;
        Collider(int entX, int entY, int xOffset, int yOffset, int width, int height);
        void shiftPosition(int xPos, int yPos);
    };
}
#endif