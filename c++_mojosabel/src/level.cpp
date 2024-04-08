#include "Level.h"
#include "FloorTile.h"
#include "WallTile.h"
#include "Session.h"

namespace mojosabel{
    
    void Level::drawTiles()
    {
        for(int x = 0; x < MAP_WIDTH; x++)
        {
            for(int y = 0; y < MAP_HEIGHT; y++)
            {
                tilemap(x, y)->getTopLayer()->draw();
            }
        }
    }

    void Level::easyFill(int x, int y, std::string tag) 
    {
        if(tag == "Floor")
        {
            tilemap(x, y)->add(new FloorTile(x * TILE_SIZE, y * TILE_SIZE, walkableTexturePath));
        } else if(tag == "Wall")
        {
            WallTile* wallTile = new WallTile(x * TILE_SIZE, y * TILE_SIZE, unwalkableTexturePath);
            tilemap(x, y)->add(wallTile);
            ses.add(wallTile);
        }
    }

    void Level::checkForFloor(){
        for(int x = 0; x < MAP_WIDTH; x++)
        {
            for(int y = 0; y < MAP_HEIGHT; y++)
            {
                if(tilemap(x, y)->getTopLayer()->tag == "Floor"){
                    
                    walkableTiles.push_back(Vector2(x * TILE_SIZE, y * TILE_SIZE));
                }
            }
        }
    }

    bool Level::isTileWall(int x, int y) {
        if ((x/TILE_SIZE >= MAP_WIDTH - 1 || y > SCREEN_HEIGHT || x/TILE_SIZE < 1 || y < 1)){  
            return true;
        } else if (tilemap(x/TILE_SIZE, y/TILE_SIZE)->getTopLayer()->tag == "Wall"){
            return true;
        } else {
            return false;
        }
    }

    const Vector2 Level::generateSpawnPosition() const
    {  
        int i = rand() % walkableTiles.size();
        Vector2 vec = walkableTiles[i];
        return vec;
    }

    void Level::clearWallPointers()
    {
        for(int x = 0; x < MAP_WIDTH; x++)
        {
            for(int y = 0; y < MAP_HEIGHT; y++)
            {
                if (tilemap(x, y)->getTopLayer()->tag == "Wall") { tilemap(x, y)->clear(); }
            }
        }      
    }
}