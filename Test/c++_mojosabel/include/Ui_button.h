#ifndef UI_BUTTON_H
#define UI_BUTTON_H

#include "Ui_sprite.h"

namespace mojosabel {
    
    class Ui_button : public Ui_sprite 
    {
        private:
            bool isDown = false;
            SDL_Texture* buttonImageDown;
            SDL_Texture* buttonImageUp;
        protected:
            Ui_button(int x, int y, int w, int h, std::string txt);
        
        public:
            static Ui_button* getInstance(int x, int y, int w, int h, std::string txt);
            void mouseDown(const SDL_Event&);
            void mouseUp(const SDL_Event&);
            void draw();
            virtual void perform(Ui_button* source) {}
            ~Ui_button();
    };
}

#endif