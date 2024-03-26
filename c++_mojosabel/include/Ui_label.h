#ifndef UI_LABEL_H
#define UI_LABEL_H

#include "Ui_sprite.h"

namespace mojosabel {

	class Ui_label: public Ui_sprite
	{
	protected:
		Ui_label(int x, int y, int w, int h, std::string txt);
	public:
		static Ui_label* getInstance(int x, int y, int w, int h, std::string txt);
		void draw();
		~Ui_label();
	};

}
#endif