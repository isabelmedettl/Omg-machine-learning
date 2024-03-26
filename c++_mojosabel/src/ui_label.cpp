#include "Ui_label.h"
#include "System.h"

using namespace std;

namespace mojosabel {

	Ui_label* Ui_label::getInstance(int x, int y, int w, int h, std::string txt) 
	{
		return new Ui_label(x, y, w, h, txt);
	}

	Ui_label::Ui_label(int x, int y, int w, int h, std::string txt): Ui_sprite(x,y,w,h,txt)
	{
		SDL_Surface* surf = TTF_RenderText_Solid(sys.getFont(), text.c_str(), { 0,0,0 });
		texture = SDL_CreateTextureFromSurface(sys.getRen(), surf);
		SDL_FreeSurface(surf);
	}

	Ui_label::~Ui_label()
	{
		SDL_DestroyTexture(texture);
	}

	void Ui_label::draw()
	{
		SDL_RenderCopy(sys.getRen(), texture, NULL, &getRect());
	}
}