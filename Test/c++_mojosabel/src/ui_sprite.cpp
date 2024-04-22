#include <string>
#include "Ui_sprite.h"
#include "System.h"
#include "Constants.h"



namespace mojosabel {

    Ui_sprite::Ui_sprite(int x, int y, int w, int h, std::string txt) : Sprite(x,y,w,h), text(txt)
    {
        SDL_Surface* surf = TTF_RenderText_Solid(sys.getFont(), text.c_str(), {0,0,0});
        texture = SDL_CreateTextureFromSurface(sys.getRen(), surf);
        SDL_FreeSurface(surf);
    }

    Ui_sprite::~Ui_sprite()
    {
        SDL_DestroyTexture(texture);
    }

	void Ui_sprite::setText(std::string newText) 
    {
		text = newText;
		SDL_DestroyTexture(texture);
		SDL_Surface* surf = TTF_RenderText_Solid(sys.getFont(), text.c_str(), { 0,0,0 });
		texture = SDL_CreateTextureFromSurface(sys.getRen(), surf);
		SDL_FreeSurface(surf);
	}
}