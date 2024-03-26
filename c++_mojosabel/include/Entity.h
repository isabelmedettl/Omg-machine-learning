#ifndef ENTITY_H
#define ENTITY_H

#include <SDL2/SDL.h>
#include <string>
#include <vector>

#include "Collision.h"
#include "Collider.h"
#include "Vector2.h"

namespace mojosabel {

    class Entity 
    {
    private:
        int delay = 60;

    protected:
        Entity() : Entity(0, 0, 0, 0, 0, "Unknown") {}
        Entity(int xPos, int yPos, int width, int height, int layer, std::string tag);
        int layer;
        SDL_Rect rect;
        bool hasCollision = false;
        SDL_Texture* texture;
        std::vector<Collider> colliders;

    public:
        Entity(const Entity* other) = delete;
        const Entity operator=(const Entity& rhs) = delete;
        const std::string tag;
        void setCollision(const bool toSet);
        bool hasColliders();
        const bool getCollision() const { return hasCollision; } 
        void addCollider(int xOffset, int yOffset, int colWidth, int colHight);
        void adjustColliders();
        int getLayer() const { return layer; }
        SDL_Rect* getRect() { return &rect; }
        std::vector<Collider>& getColliders();
        Vector2 position() const;
        void setPosition(const Vector2 vecToSet);
        void loadTexture(const std::string filename);
        void resizeToImage();
        void draw() const;
        void sneakyUpdate();
        virtual void start() {}
        virtual void update() {}
        virtual void fixedUpdate() {}
        virtual void onCollision(Collision<Entity> collision) {} 
        virtual void mouseUp(int posX, int posY) {}
        virtual void mouseDown(int posX, int posY) {}
        virtual ~Entity();
    };
}

#endif