#ifndef LEVEL_H
#define LEVEL_H

#include "Tilemap.h"


namespace mojosabel {
class Level {
    private:
        Tilemap tilemap;
        std::string walkableTexturePath, unwalkableTexturePath;
        std::vector<Vector2> walkableTiles;

    public:
        Level() = delete;
        Level(const Level* level) = delete;
        const Level operator=(const Level& rhs) = delete;
        Level(std::string walkableTexturePath, std::string unwalkableTexturePath) 
            : walkableTexturePath(walkableTexturePath), unwalkableTexturePath(unwalkableTexturePath) {}
        void drawTiles();
        void easyFill(int x, int y, std::string tag);
        void checkForFloor();
        bool isTileWall(int x, int y);
        const Vector2 generateSpawnPosition() const; // möjligen enemypointer 
        std::string getWalkableTexPath() { return walkableTexturePath; }
        std::string getUnwalkableTexPath() { return unwalkableTexturePath; }
        void clearWallPointers();
        Tile* getTile(int x, int y) { return tilemap(x, y); }
    };
}


#endif