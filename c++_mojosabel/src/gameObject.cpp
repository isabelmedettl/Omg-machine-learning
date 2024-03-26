#include "GameObject.h"
#include "Session.h"

namespace mojosabel
{
    void GameObject::instantiate(Entity* entity)
    {
        if (entity != this)
        {
            ses.add(entity);
        }
    }

    void GameObject::destroy(Entity* entity) 
    {
        ses.remove(entity);
    }

    void moveLeft(GameObject* gameObject)
    {
        gameObject->getRect()->x -= gameObject->getSpeed();
    }

    void moveRight(GameObject* gameObject)
    {
        gameObject->getRect()->x += gameObject->getSpeed();
    }

    void moveUp(GameObject* gameObject)
    {
        gameObject->getRect()->y -= gameObject->getSpeed();
    }

    void moveDown(GameObject* gameObject)
    {
        gameObject->getRect()->y += gameObject->getSpeed();
    }

    bool GameObject::checkDirection(const std::string direction)
    {
        Level* level = ses.getWorld()->getCurrentLevel();
        if (direction == "Up"){
            return (level->isTileWall(rect.x + 1, rect.y - speed) || level->isTileWall(rect.x - 1 + rect.w, rect.y - speed));
        }
        if (direction == "Down"){
            return (level->isTileWall(rect.x + 1, rect.y + rect.h + speed) || level->isTileWall(rect.x - 1 + rect.w, rect.y + rect.h + speed));
        }
        if (direction == "Left"){
            return (level->isTileWall(rect.x - speed, rect.y + 1) || level->isTileWall(rect.x - speed, rect.y - 1 + rect.h));
        }
        if (direction == "Right"){
            return (level->isTileWall(rect.x + rect.w + speed, rect.y + 1) || level->isTileWall(rect.x + rect.w + speed, rect.y - 1 + rect.h));
        }

        return false;
    }
} 
