#include "FloorTile.h"
#include "Constants.h"

namespace mojosabel {

    FloorTile::FloorTile(int xPos, int yPos, std::string texture) : Entity(xPos, yPos, TILE_SIZE, TILE_SIZE, -1, "Floor")
    { 
        loadTexture(constants::gResPath + texture); 
    }
}
