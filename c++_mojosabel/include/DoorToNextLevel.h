#ifndef DOORTONEXTLEVEL_H
#define DOORTONEXTLEVEL_H

#include "GameObject.h"
#include "Constants.h"

namespace mojosabel  {

    class DoorToNextLevel : public GameObject
    {
    public:
        DoorToNextLevel() = delete;
        DoorToNextLevel(int x, int y);
        void onCollision(Collision<Entity> collision);
    };
}

#endif