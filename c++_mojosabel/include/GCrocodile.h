#ifndef GCROCODILE_H
#define GCROCODILE_H

#include "Enemy.h"
#include "Player.h"

using namespace mojosabel;

class GCrocodile : public Enemy
{
private:
    bool hasPlayer = false;
    Player* playerPtr;
public:
    
    GCrocodile(int x, int y) : Enemy(x, y) { setHealth(20); }
    void update();
    void moveToPlayer();
    void checkHealth();
    //void attack();
    bool checkPlayerDistance();
    
    void setHasPlayer(bool toSet) { hasPlayer = toSet; }
    bool getHasPlayer() { return hasPlayer; }
    void setPlayer(Player* ptr) { playerPtr = ptr; }
    Player* getPlayer() { return playerPtr; }
};

#endif