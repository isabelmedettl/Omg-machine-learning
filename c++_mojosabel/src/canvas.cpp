#include <algorithm>
#include <iostream>
#include "Canvas.h"

namespace mojosabel {

    

    void Canvas::removeCanvas(Canvas* canvasToRemove)
    {
        for (std::vector<Canvas*>::iterator it = canvases.begin(); it <= canvases.end(); ++it)
        {
            if (canvasToRemove == *it)
            {
                it = canvases.erase(it);
            }
        }
    }

    void Canvas::addUiSprite(Ui_sprite* uiSpriteToAdd) 
    { 
        uiSprites.push_back(uiSpriteToAdd);
        sortSprites();
    }

    void Canvas::removeUiSprite(Ui_sprite* uiSpriteToRemove)
    {
        for (std::vector<Ui_sprite*>::iterator it = uiSprites.begin(); it != uiSprites.end(); ++it)
        {
            if (uiSpriteToRemove == *it)
            {
                it = uiSprites.erase(it);
            }
        }
    }

    void Canvas::drawSprites()
    {
        for (Ui_sprite* s : uiSprites)
        {
            s->draw();
        }

        for (Canvas* c : canvases)
        {
            c->drawSprites();
        }
    }

    void Canvas::sortSprites()
    {
        std::sort(uiSprites.begin(), uiSprites.end(), compareLayer);
    }

    void Canvas::handleMouseUp(const SDL_Event& event)
    {
        for (Ui_sprite* s : uiSprites)
        {
            s->mouseUp(event);
        }

        for (Canvas* c : canvases)
        {
            c->handleMouseUp(event);
        }
    }
    
    void Canvas::handleMouseDown(const SDL_Event& event)
    {
         for (Ui_sprite* s : uiSprites)
        {
            s->mouseDown(event);
        }
        for (Canvas* c : canvases)
        {
            c->handleMouseDown(event);
        }
    }

    Canvas::~Canvas()
    {
        for (Canvas* p : canvases) { delete p; }
        for (Ui_sprite* p : uiSprites) { delete p; }
        canvases.clear();
        uiSprites.clear();

    }
}