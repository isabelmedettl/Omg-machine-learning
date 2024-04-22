#ifndef JOSDIGGINGBULLET_H
#define JOSDIGGINGBULLET_H

#include "JosBullet.h"

class JosDiggingBullet : public JosBullet {

public:
    JosDiggingBullet(int x, int y, int targetX, int targetY) 
        : JosBullet(x, y, "DiggingBullet", 8, targetX, targetY, 1) { }
    void onCollision(Collision<Entity> collision);
};

#endif