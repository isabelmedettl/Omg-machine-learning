#ifndef GTURTLE_H
#define GTURTLE_H

#include "Player.h"
#include "Ui_label.h"

using namespace mojosabel;

class GTurtle : public Player
{
    private:
        int iFrameCounter = 60;
        bool canShoot;
        bool isAlive;
        int shootCoolDown = 6;
        int shootCounter = 0;
    public:
        GTurtle(int x, int y, int startHealth);
        void adjustHealth(int changeHealth);
        void onCollision(Collision<Entity> collision);
        void fire(int x, int y);
        void die();
        void update();
        bool levelCompleted();
        void resetForNewLevel();
};

#endif