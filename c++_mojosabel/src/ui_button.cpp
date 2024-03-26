#include <SDL2/SDL_image.h>

#include "Ui_button.h"
#include "System.h"
#include "Constants.h"


using namespace std;

namespace mojosabel {

	Ui_button* Ui_button::getInstance(int x, int y, int w, int h, std::string txt) 
	{
		return new Ui_button(x, y, w, h, txt);
	}

	Ui_button::Ui_button(int x, int y, int w, int h, std::string txt): Ui_sprite(x,y,w,h,txt)
	{
		SDL_Surface* surf = TTF_RenderText_Solid(sys.getFont(), text.c_str(), { 0,0,0 });
		texture = SDL_CreateTextureFromSurface(sys.getRen(), surf);
		SDL_FreeSurface(surf);
		buttonImageDown = IMG_LoadTexture(sys.getRen(), (constants::gResPath + "images/DownButton.png").c_str() );
		buttonImageUp = IMG_LoadTexture(sys.getRen(), (constants::gResPath + "images/UpButton.png").c_str() );
	}

	Ui_button::~Ui_button()
	{
		SDL_DestroyTexture(texture);
		SDL_DestroyTexture(buttonImageDown);
		SDL_DestroyTexture(buttonImageUp);
	}
	

	void Ui_button::draw() 
	{	
		if (isDown)
		{
			SDL_RenderCopy(sys.getRen(), buttonImageDown, NULL, &getRect());
		}
		else 
		{
			SDL_RenderCopy(sys.getRen(), buttonImageUp, NULL, &getRect());
		
		}
		SDL_RenderCopy(sys.getRen(), texture, NULL, &getRect());
	}

	void Ui_button::mouseDown(const SDL_Event& event)
	{
		SDL_Point p = { event.button.x, event.button.y };
		if (SDL_PointInRect(&p, &getRect()))
		{
			isDown = true;
		}
	}

	void Ui_button::mouseUp(const SDL_Event& event)
	{
		SDL_Point p = { event.button.x, event.button.y };
		if (SDL_PointInRect(&p, &getRect()))
		{
			perform(this);
		}
		isDown = false;
	}
}