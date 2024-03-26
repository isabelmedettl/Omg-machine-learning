#ifndef GAMEOBJECT_H
#define GAMEOBJECT_H

#include "Entity.h"

namespace mojosabel {

    class GameObject : public Entity
    {    
    protected:
        int speed;
        GameObject() : Entity(0, 0, 0, 0, 0, "Unkown") {}
        GameObject(int x, int y, int w, int h, int layer, std::string tag) 
            : Entity(x, y, w, h, layer, tag) {}
        void instantiate(Entity* entity);
    public:
        void destroy(Entity* entity);
        bool checkDirection(const std::string direction);
        int getSpeed() const { return speed; }
    };

    void moveLeft(GameObject* gameObject);
    void moveRight(GameObject* gameObject);
    void moveUp(GameObject* gameObject);
    void moveDown(GameObject* gameObject);
}

#endif