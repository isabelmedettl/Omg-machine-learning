#ifndef WALLTILE_H
#define WALLTILE_H

#include "Entity.h"
#include "Constants.h"

namespace mojosabel{

    class WallTile : public Entity
    {
    public:
        WallTile(int xPos, int yPos, std::string texture);
        WallTile() = delete;
    };
}
#endif