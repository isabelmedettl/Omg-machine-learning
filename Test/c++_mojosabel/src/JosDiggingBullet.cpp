#include "JosDiggingBullet.h"
#include "FloorTile.h"
#include "Session.h"

void JosDiggingBullet::onCollision(Collision<Entity> collision) {

    if (collision.tag == "Wall") {
        Entity* temp = new FloorTile(collision.object->getRect()->x, collision.object->getRect()->y, ses.getWorld()->getCurrentLevel()->getWalkableTexPath());
        destroy(collision.object);
        ses.getWorld()->getCurrentLevel()->getTile(temp->getRect()->x / TILE_SIZE, temp->getRect()->y / TILE_SIZE)->setTopLayer(temp);
        destroy(this);
    }
}