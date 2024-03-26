#ifndef SPRITE_H
#define SPRITE_H

#include <SDL2/SDL.h>
#include <string>

namespace mojosabel {

    class Sprite {
    private:
        int layer = 0;
        // SDL_Rect rect;
        Sprite(const Sprite&) = delete;
        const Sprite& operator=(const Sprite&) = delete;
    protected:
        SDL_Rect rect;
        Sprite(int x, int y, int w, int h) : rect {x, y, w, h} {}
    public:
        virtual ~Sprite() {}
        virtual void mouseDown(const SDL_Event&) {}
		virtual void mouseUp(const SDL_Event&) {}
        virtual void keyDown(const SDL_Event&) {}
		virtual void keyUp(const SDL_Event&) {}
        virtual void draw() = 0;
        //virtual void tick();
        const SDL_Rect& getRect() const { return rect; };
        int getLayer() const { return layer; }
    };

    bool compareLayer(Sprite*, Sprite*);
}



#endif