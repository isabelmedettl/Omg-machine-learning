#ifndef UI_SPRITE_H
#define UI_SPRITE_H

#include "Sprite.h"


namespace mojosabel {

    class Ui_sprite : public Sprite 
    {  
    protected:
        std::string text;
        SDL_Texture* texture;
        Ui_sprite(int x, int y, int w, int h, std::string txt);
    public:
        void setText(std::string text);
        std::string getText() const { return text; }
        virtual ~Ui_sprite();
    };
}


#endif