#include "Sprite.h"

namespace mojosabel {

   
    bool compareLayer(Sprite* s1, Sprite* s2) {
        return s1->getLayer() < s2->getLayer();
    }
}