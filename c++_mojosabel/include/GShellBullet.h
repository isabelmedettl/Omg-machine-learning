#ifndef GSHELLBULLET_H
#define GSHELLBULLET_H

#include "Bullet.h"

using namespace mojosabel;

class GShellBullet : public Bullet
{
private:
    int damage = 5;
public:
    GShellBullet(int x, int y, int targetX, int targetY) : Bullet(x, y, 16, 16, 0, "Bullet", 30, targetX, targetY) {}
    void onCollision(Collision<Entity> collision);
};

#endif