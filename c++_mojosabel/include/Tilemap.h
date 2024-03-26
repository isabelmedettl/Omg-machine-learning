#ifndef TILEMAP_H
#define TILEMAP_H

#include "Tile.h"
#include "Constants.h"

namespace mojosabel
{
    struct Tilemap
    {
    private:
        Tile* map[MAP_WIDTH][MAP_HEIGHT];
    
    public:
        Tilemap();
        Tilemap(const Tilemap* other) = delete;
        const Tilemap operator=(const Tilemap& rhs) = delete;
        Tile* operator()(int x, int y){return map[x][y];}
        ~Tilemap();
    };
}

#endif