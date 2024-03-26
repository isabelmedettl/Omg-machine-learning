#include <string>
#include "Session.h"
#include "World.h"
#include "GameObjectGenerator.h"
#include "DoorToNextLevel.h"

namespace mojosabel {
    
    World::World(int smoothMap, int fillPercent, int smoothWalkableLimit, int smoothUnwalkableLimit)
    {
        mapGenerator = new BitwiseMapGenerator(smoothMap, fillPercent, smoothWalkableLimit, smoothUnwalkableLimit);
    }

    void World::newLevel(std::string walkableTexturePath, std::string unwalkableTexturePath)
    {
        if (levels.size() == 0)
        {
            currentLevel = 0;
        }
        else 
        {
            ses.clearEntitiesExcept("Player");
            currentLevel++;
        }
        Level* level = new Level(walkableTexturePath, unwalkableTexturePath); 
        levels.push_back(level);
        mapGenerator->generateBitMap();
        mapGenerator->mixMap(level);
        if (Entity* player = ses.findEntity("Player"))
        {
            player->setPosition(level->generateSpawnPosition());
        }
        generateGameObjects<DoorToNextLevel>(level, 1, "images/Door.png", true);
    }

    void World::drawCurrentLevel()
    {
        levels[currentLevel]->drawTiles();
    }

    World::~World()
    {
        for (Level* p : levels) { delete p; }
        levels.clear();
        delete mapGenerator;
    }
}