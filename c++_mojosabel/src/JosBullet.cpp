#include "JosBullet.h"

JosBullet::JosBullet(int posX, int posY, std::string tag, int speed, int targetX, int targetY, int damage) 
        : Bullet(posX, posY, 16, 16, 1, tag, speed, targetX, targetY), damage(damage) 
{
    loadTexture(constants::gResPath + "images/DrillBullet32p.png");
    setCollision(true);
}