#ifndef CANVAS_H
#define CANVAS_H

#include <vector>
#include "Ui_sprite.h"

namespace mojosabel {
    
    class Canvas 
    {
    private:
        int xPos;
        int yPos;
        int width;
        int height;
        std::vector<Canvas*> canvases;
        std::vector<Ui_sprite*> uiSprites;
    public:
        Canvas() : Canvas(0, 0, 0, 0) {}
        Canvas(int x, int y, int w, int h) { xPos = x, yPos = y, width = w, height = h; }
        Canvas(const Canvas* other) = delete;
        const Canvas operator=(const Canvas& rhs) = delete;
        int x() { return xPos; }
        int y() { return yPos; }
        int w() { return width; }
        int h() { return height; }
        void addCanvas(Canvas* canvasToAdd) { if (canvasToAdd != this) {canvases.push_back(canvasToAdd);} }
        void addUiSprite(Ui_sprite* uiSpriteToAdd);
        void removeCanvas(Canvas* canvasToRemove);
        void removeUiSprite(Ui_sprite* uiSpriteToRemove);
        void handleMouseUp(const SDL_Event& event);
        void handleMouseDown(const SDL_Event& event);
        void sortSprites();
        void drawSprites();
        ~Canvas();
    };

}

#endif