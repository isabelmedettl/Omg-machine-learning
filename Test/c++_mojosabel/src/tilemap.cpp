#include "Tilemap.h"

namespace mojosabel{

    Tilemap::Tilemap()
    {
        for(int x = 0; x < MAP_WIDTH; x++)
        {
            for(int y = 0; y < MAP_HEIGHT; y++)
            {
                map[x][y] = new Tile;
            }
        }
    }

    Tilemap::~Tilemap()
    {
        for(int x = 0; x < MAP_WIDTH; x++)
        {
            for(int y = 0; y < MAP_HEIGHT; y++)
            {
                delete map[x][y]; 
            }
        }
    }
}