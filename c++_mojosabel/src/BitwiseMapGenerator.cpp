#include "BitwiseMapGenerator.h"

namespace mojosabel {
    void BitwiseMapGenerator::generateBitMap()
    {
        randomFillBitMap();
        for(int i = 0; i < smc; i++)
        {
            smoothMap();
        }
    }

    void BitwiseMapGenerator::randomFillBitMap(){
        for(int x = 0; x < MAP_WIDTH; x++)
        {
            for(int y = 0; y < MAP_HEIGHT; y++)
            {
                int i = rand() % 20 + 40;
                if(i < rfp){ bitMap[y] |= 1UL << x; }
                else{ bitMap[y] &= ~(1UL << x); }
            }
        }
    }

    void BitwiseMapGenerator::mixMap(Level* level){
        for(int x = 0; x < MAP_WIDTH; x++)
        {
            for(int y = 0; y < MAP_HEIGHT; y++)
            {
                //kolla nere y
                if (y == Y_MIN_OFFSET || y == Y_MAX_OFFSET || x <= X_MIN_OFFSET || x >= X_MAX_OFFSET)
                {
                    level->easyFill(x, y, "Wall");
                }
                else if (bitMap[y] & (1UL << x))
                {
                    level->easyFill(x, y, "Wall");
                } 
                else 
                {
                    level->easyFill(x, y, "Floor");
                }
            }
        }
        level->checkForFloor();
    }

    int BitwiseMapGenerator::getSurroundingWallCount(int tileX, int tileY){
        int wallCount = 0;
        for(int neighbourX = tileX - 1; neighbourX <= tileX + 1; neighbourX ++)
        {
            for(int neighbourY = tileY - 1; neighbourY <= tileY + 1; neighbourY ++)
            {
                if (neighbourX >= 0 && neighbourX < MAP_WIDTH && neighbourY >= 0 && neighbourY < MAP_HEIGHT)
                {
                    if (neighbourX != tileX || neighbourY != tileY)
                    {
                        if(bitMap[neighbourY] & (1UL << neighbourX)){wallCount++;}
                    }
                } 
                else 
                {
                    wallCount++;
                }
            }
        }
        return wallCount;
    }

    void BitwiseMapGenerator::smoothMap()
    {
        for(int x = 0; x < MAP_WIDTH; x++)
        {
            for(int y = 0; y < MAP_HEIGHT; y++)
            {
                int neighbourWalls = getSurroundingWallCount(x,y);
                if(neighbourWalls > sul)
                {
                    bitMap[y] |= (1UL << x);
                } 
                else if (neighbourWalls < swl)
                {
                    bitMap[y] &= ~(1UL << x);
                }
            }
        }
    }
}