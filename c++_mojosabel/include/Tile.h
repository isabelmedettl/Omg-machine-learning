#ifndef TILE_H
#define TILE_H

#include <vector>
#include "Entity.h"

namespace mojosabel {

   class Tile 
   {
    private:
        std::vector<Entity*> tileObjects;
        void sort();
    public:
        Tile() {}
        Tile(const Tile* other) = delete;
        const Tile operator=(const Tile& rhs) = delete;
        Entity* getTopLayer() { return tileObjects[0]; } // när vi ritar ut tiles i tilemap så hämtar vi det översta lagret och ritar ut det.
        void add(Entity* entityToAdd);
        void remove(Entity* entityToRemove);
        void clear() { tileObjects.clear(); }
        ~Tile();
        void setTopLayer(Entity* entity) { tileObjects[0] = entity; }
    }; 
    bool compareLayer(Entity* e1, Entity* e2);
}

#endif

