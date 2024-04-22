#ifndef BULLET_H
#define BULLET_H

#include "GameObject.h"
#include "Vector2.h"

namespace mojosabel {

    class Bullet : public GameObject
    {
    protected:
        Vector2 target;
    public:
        Bullet() = delete;
        Bullet(int x, int y, int w, int h, int layer, std::string tag, int speed, int targetX, int targetY);
        virtual void update();
        virtual void move();
        virtual void onCollision(Collision<Entity> collision);
    };
}
#endif