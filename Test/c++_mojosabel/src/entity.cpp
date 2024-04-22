#include <SDL2/SDL_image.h>
#include <iostream>
#include "System.h"
#include "Entity.h"


namespace mojosabel {

    Entity::Entity(int xPos, int yPos, int width, int height, int layer, std::string tag) 
        : layer(layer), tag(tag) 
    {
        rect.x = xPos;
        rect.y = yPos;
        rect.w = width;
        rect.h = height;
    } 
        
    void Entity::loadTexture(const std::string filename)
    {
        texture = IMG_LoadTexture(sys.getRen(), (filename).c_str());
    }

    void Entity::resizeToImage()
    {
        SDL_QueryTexture(texture, NULL, NULL, &rect.w, &rect.h);
    }

    void Entity::draw() const
    {
        if(texture)
        {
            SDL_RenderCopy(sys.getRen(), texture, NULL, &rect);
        }
    }

    void Entity::sneakyUpdate()
    {
        update();
        draw();
        adjustColliders();
        delay--;
        if (delay <= 0) 
        {
            fixedUpdate();
            delay = 60;
        }
    }

    void Entity::setCollision(const bool toSet)
    {
        hasCollision = toSet;
    }

    bool Entity::hasColliders()
    {
        return (colliders.size());
    }
    
    void Entity::addCollider(int xOffset, int yOffset, int colWidth, int colHight)
    {
        Collider collider = Collider(rect.x, rect.y, xOffset, yOffset, colWidth, colHight);
        colliders.push_back(collider);
    }
    
    void Entity::adjustColliders()
    {
        for (Collider c : colliders)
        {
           c.shiftPosition(rect.x, rect.y);
        }
    }
    
    std::vector<Collider>& Entity::getColliders()
    {
        return colliders;
    }

    Vector2 Entity::position() const
    {
        return Vector2{rect.x, rect.y}; 
    }
    
    void Entity::setPosition(const Vector2 vecToSet)
    {
        rect.x = vecToSet.x;
        rect.y = vecToSet.y;
    }

    Entity::~Entity() 
    { 
        if(texture) { SDL_DestroyTexture(texture); }
    }
}