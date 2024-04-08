#ifndef WORLD_H
#define WORLD_H

#include <vector>
#include "Level.h"
#include "BitwiseMapGenerator.h"


namespace mojosabel {
    
    class World
    {
    private:
        int currentLevel;
        int maxProgression = 3;
        BitwiseMapGenerator* mapGenerator;
        std::vector<Level*> levels;
        
    public:
        World() : World(3, 49, 4, 4) {}
        World(const World* other) = delete;
        const World operator=(const World& rhs) = delete;
        World(int smoothMap, int fillPercent, int smoothWalkableLimit, int smoothUnwalkableLimit);
        void newLevel(std::string walkableTexturePath, std::string unwalkableTexturePath);
        void drawCurrentLevel();
        const int getCurrentLevelIndex() { return currentLevel; }
        Level* getCurrentLevel() { return levels[currentLevel]; }
        BitwiseMapGenerator* getMapGenerator() { return mapGenerator; }
        void setMaxProgression(int newMaxProgression);
        int getMaxProgression() { return maxProgression; }
        int currentProgressionValue = 0;
        ~World();
    };
}

#endif