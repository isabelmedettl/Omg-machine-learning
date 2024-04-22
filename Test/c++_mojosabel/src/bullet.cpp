#include <iostream>
#include "Bullet.h"
#include "System.h"
#include "Session.h"


namespace mojosabel {
    
    Bullet::Bullet(int x, int y, int w, int h, int layer, std::string tag, int s, int targetX, int targetY)
        : GameObject(x, y, w, h, layer, tag)
    {
        // göra en medlem som sköter nedan
        speed = s;
        Vector2 targetPosition = Vector2{targetX, targetY};
        targetPosition = targetPosition - position();
        targetPosition = targetPosition * 100;
        targetPosition = targetPosition + position();
        target = targetPosition;
    }

    void Bullet::update()
    {
        move();

        if (sys.isOutOfBounds(rect.x, rect.y))
        {
            destroy(this);
        }
    }

    void Bullet::move()
    {
        setPosition(position().moveTowards(target, speed));
    }
    
    void Bullet::onCollision(Collision<Entity> collision)
    {
        if (collision.tag == "Enemy") 
        { 
            std::cout << "hit enemy" << std::endl; 
            destroy(collision.object);
            destroy(this);
        }
        if (collision.tag == "Wall")
        {
            destroy(this);
        }
    }

}