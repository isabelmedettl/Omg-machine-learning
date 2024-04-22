#include "DoorToNextLevel.h"
#include "Player.h"
#include "Session.h"
#include <iostream>

namespace mojosabel {

    DoorToNextLevel::DoorToNextLevel(int x, int y) : GameObject(x, y, TILE_SIZE, TILE_SIZE, 0, "Door") 
    {
        loadTexture(constants::gResPath + "images/Beaver.png");         
    }
    void DoorToNextLevel::onCollision(Collision<Entity> collision)
    {
        if (collision.tag == "Player")
        {
            Player* player = static_cast<Player*>(collision.object);
            if (player->levelCompleted())
            {   
                std::cout << "level completed" << std::endl;
                ses.setLoadNextLevel(true);
            }
        }
    }
}
