#ifndef ENEMY_H
#define ENEMY_H

#include "GameObject.h"

namespace mojosabel {

    class Enemy : public GameObject
    {
    protected:
        int health;
    public:
        Enemy() = delete;
        Enemy(int x, int y);
        Enemy(int x, int y, int w, int h, int layer, std::string name, int speed);
        int getHealth() { return health; }
        void setHealth(int healthToSet) {health = healthToSet; }
        virtual void update();
        virtual void move();
        typedef void (*moveFunc)(GameObject* gameObject);
        moveFunc mF;
    };
}

#endif