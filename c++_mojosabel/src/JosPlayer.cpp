#include "JosPlayer.h"
#include "JosDiggingBullet.h"
#include "Constants.h"

JosPlayer::JosPlayer(int x, int y) : Player(x, y, 32, 32, 3, 4) {
    setCollision(true);
    loadTexture(constants::gResPath + "images/DesertDigger32p.png");
}

void JosPlayer::fire(int x, int y)
{
    JosDiggingBullet *bullet = new JosDiggingBullet(rect.x, rect.y, x, y);
    instantiate(bullet);
}

bool JosPlayer::levelCompleted(){
    if(crystalsRequired <= 0) { return true; }
    return false;
}

void JosPlayer::onCollision(Collision<Entity> collision){
    if(collision.tag == "Pickup"){
        crystalsRequired--;
        destroy(collision.object);
    }
}