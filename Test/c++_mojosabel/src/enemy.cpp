#include <iostream>
#include "Enemy.h"
#include "System.h"

namespace mojosabel {

    Enemy::Enemy(int x, int y) : Enemy(x, y, 32, 32, 0, "Enemy", 1)
    {
        
    }

    Enemy::Enemy(int x, int y, int w, int h, int layer, std::string name, int sp)
        : GameObject (x, y, w, h, layer, name)
    {
        speed = sp;
        mF = moveLeft;
    }

    void Enemy::update()
    {
        move();
    }

  
    void Enemy::move()
    {
        if(checkDirection("Left")){
            mF = moveRight;
        } else if(checkDirection("Right")){
            mF = moveLeft;
        }

        mF(this);
    }
}