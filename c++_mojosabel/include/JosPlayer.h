#ifndef JOSPLAYER_H
#define JOSPLAYER_H

#include "Player.h"

using namespace mojosabel;

class JosPlayer : public Player {
private:
    int crystalsRequired;
public:
    JosPlayer(int x, int y);
    void fire(int x, int y);
    bool levelCompleted();
    void onCollision(Collision<Entity> collision);
    void setCrystalsRequired(int req) { crystalsRequired = req; }
};
#endif