#include <algorithm>
#include "Tile.h"

namespace mojosabel {

    void Tile::add(Entity* entityToAdd)
    {
        tileObjects.push_back(entityToAdd);
        sort();
    }

    void Tile::remove(Entity* entityToRemove)
    {
        for (std::vector<Entity*>::iterator it = tileObjects.begin(); it != tileObjects.end(); ++it)
        {
            if (*it == entityToRemove)
            {
                it = tileObjects.erase(it);
            }
        }
    }

    void Tile::sort()
    {
        std::sort(tileObjects.begin(), tileObjects.end(), compareLayer);
    }

    bool compareLayer(Entity* e1, Entity* e2) 
    {
        return e1->getLayer() < e2->getLayer();
    }

    Tile::~Tile()
    {
        for(Entity* p : tileObjects) 
        { 
           delete p;  
        }
        tileObjects.clear();
    }
}