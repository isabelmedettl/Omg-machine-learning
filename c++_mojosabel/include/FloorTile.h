#ifndef FLOORTILE_H
#define FLOORTILE_H

#include "Entity.h"

namespace mojosabel{

    class FloorTile : public Entity
    {
    public:
        FloorTile(int xPos, int yPos, std::string texture);
        FloorTile() = delete;
    };
}
#endif