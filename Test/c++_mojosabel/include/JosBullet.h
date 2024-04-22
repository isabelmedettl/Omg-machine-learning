#ifndef JOSBULLET_H
#define JOSBULLET_H

#include "Bullet.h"
#include "Constants.h"

using namespace mojosabel;

class JosBullet : public Bullet {
private:
    int damage;
public:
    JosBullet(int posX, int posY, std::string tag, int speed, int targetX, int targetY, int damage);
};

#endif