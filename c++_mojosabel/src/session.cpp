#include <iostream>
#include <algorithm>
#include <SDL2/SDL.h>
#include "Session.h"
#include "System.h"
#include "Collision.h"

namespace mojosabel {

    void Session::saveRenderedImage()
    {
        std::cout << "printy" << std::endl;

    }

    void Session::compareSurfaces()
    {
        std::cout << "compary" << std::endl;
    }

    Session::Session()
    {
        std::cout << "Hej det funkar, session" << std::endl;
        rootCanvas = new Canvas();
        //world = new World();
    }

    void Session::add(Entity* entityToAdd)
    {
        addedEntities.push_back(entityToAdd);
    }

    void Session::remove(Entity* entityToRemove)
    {
        removedEntities.push_back(entityToRemove); // göra en removed ses i entity?
    } 

    void Session::doKeyUp(SDL_KeyboardEvent *event)
    {
        if (event->keysym.scancode < MAX_KEYBOARD_KEYS)
        {
            sys.keyboard[event->keysym.scancode] = 0;
        }
    }

    void Session::doKeyDown(SDL_KeyboardEvent *event)
    {
        if (/*event->repeat == 0 && */ event->keysym.scancode < MAX_KEYBOARD_KEYS)
        {
            sys.keyboard[event->keysym.scancode] = 1;
        }
    }

    void Session::capFrameRate (long *renderTime, float *remainder)
    {
        long wait, frameTime;

        wait = 16 + *remainder; 

        *remainder -= (int)*remainder;

        frameTime = SDL_GetTicks() - *renderTime;

        wait -= frameTime;

        if (wait < 1) { wait = 1; }

        SDL_Delay(wait);

        *remainder += 0.667;

        *renderTime = SDL_GetTicks();
    }

    bool Session::entityExists(Entity* entity)
    {
        for (Entity* e : entities)
        {
            if (e == entity) { return true; }
        }
        return false;
    }

    Entity* Session::findEntity(std::string name)
    {
        for (Entity* e : entities)
        {
            if (e->tag == name) { return e; }
        }
        return nullptr;
    }

    void Session::checkAllCollisions(Entity* entityToCheck)
    { 
        if (!entityToCheck->getCollision()) { return; }
        
        for (Entity* entity : entities) // för varje entities i sessions vec
        {
            if (entity != entityToCheck && entity->getCollision()) // om entityn vi är på inte är det vi vill kolla och har collision
            {
                if (checkCollision(*entityToCheck->getRect(), *entity->getRect())) // kollar först om objektens rects kolliderar
                {   
                    if (entityToCheck->hasColliders() && entity->hasColliders()) // om båda har colliders: jämför båda objektens colliders med varje av den andres colliders
                    {
                       for (Collider c : entityToCheck->getColliders()) 
                       {
                            if (checkColliders(c.rect, entity->getColliders())) // om någon av colliders kolliderar: skapa en collision och kör on collision i objektet vi kollar
                            {
                                Collision<Entity> col = Collision(entity, entity->tag);
                                entityToCheck->onCollision(col);
                            } 
                       }
                    }
                    else if (!entityToCheck->hasColliders() && entity->hasColliders()) // om objektet vi är på har fler colliders: kolla objektet vi vill kollas rect med varje collider i objektet vi är på
                    {
                        if (checkColliders(*entityToCheck->getRect(), entity->getColliders())) 
                        {
                            Collision<Entity> col = Collision(entity, entity->tag);
                            entityToCheck->onCollision(col);
                        }
                    }
                    else if (entityToCheck->hasColliders() && !entity->hasColliders()) // om objektet vi vill kolla har fler colliders men objektet vi är på inte har det, jämför 
                    {
                        if (checkColliders(*entity->getRect(), entityToCheck->getColliders()))
                        {
                            Collision<Entity> col = Collision(entity, entity->tag);
                            entityToCheck->onCollision(col);
                        }
                    }
                    else 
                    {
                        Collision<Entity> col = Collision(entity, entity->tag);
                        entityToCheck->onCollision(col);
                    }
                }
            }
        }
    }

    bool Session::checkColliders(SDL_Rect rectToCheck, std::vector<Collider>& colliders)
    {
        for (Collider c : colliders)
        {
            if (checkCollision(rectToCheck, c.rect))
            {
                return true;
            }
        }
        return false;
    }

    bool Session::checkCollision(SDL_Rect rect1, SDL_Rect rect2)
    {
        float rect1Left, rect2Left, rect1Right, rect2Right, rect1Top, rect2Top, rect1Bottom, rect2Bottom;

        // Calculate rect1 sides
        rect1Left = rect1.x;
        rect1Right = rect1.x + rect1.w;
        rect1Top = rect1.y;
        rect1Bottom = rect1.y + rect1.h;

        //Calculate rect2 sides
        rect2Left = rect2.x;
        rect2Right = rect2.x + rect2.w;
        rect2Top = rect2.y;
        rect2Bottom = rect2.y + rect2.h;

        //If any side from rect1 are outsied of rect2
        if (rect1Bottom <= rect2Top) { return false; }
        if (rect1Top >= rect2Bottom) { return false; }
        if (rect1Right <= rect2Left) { return false; }
        if (rect1Left >= rect2Right) { return false; }
        
        // if none are 
        return true;
    }

    void Session::createNewWorld(int smoothMap, int fillPercent, int smoothWalkableLimit, int smoothUnwalkableLimit)
    {
        world = new World(smoothMap, fillPercent, smoothUnwalkableLimit, smoothUnwalkableLimit);
    }

    void Session::run()
    {
        std::cout << "Session started" << std::endl;
        renderTime = SDL_GetTicks();
        remainder = 0;
        bool quit = false;
        int saveImageCounter = 0;
        while(!quit)
        {
            SDL_Event event;
            while(SDL_PollEvent(&event))
            {
                switch(event.type)
                {
                    case SDL_QUIT:
                        quit = true;
                        break;
                    case SDL_MOUSEBUTTONDOWN:
                        rootCanvas->handleMouseDown(event);
                        for (Entity* e : entities)
                        {
                            e -> mouseDown(event.button.x, event.button.y);
                        }
                        break;
                    case SDL_MOUSEBUTTONUP:
                        rootCanvas->handleMouseUp(event);
                        for (Entity* e : entities)
                        {
                            e -> mouseUp(event.button.x, event.button.y);
                        }
                        break;
                    case SDL_KEYDOWN:
                        doKeyDown(&event.key);
                        break;
                    case SDL_KEYUP:
                        doKeyUp(&event.key);
                        break;
                }
            }

            
            SDL_SetRenderDrawColor(sys.getRen(), 255, 255, 255, 255);
            SDL_RenderClear(sys.getRen());
            world->drawCurrentLevel();

            // Uppdaterar och ritar entities
            for (Entity* e : entities)
            {
                e->sneakyUpdate();
                checkAllCollisions(e);
                
            }
 
            for (Entity* e : addedEntities)
            {
                entities.push_back(e);
                sortEntitiesByLayer();
            }
            addedEntities.clear();


            for (Entity* e : removedEntities)
            {
                for (std::vector<Entity*>::iterator it = entities.begin(); it != entities.end();)
                {
                    if (*it == e)
                    {
                        delete e;
                        it = entities.erase(it);
                    }
                    else 
                    {
                        it++;
                    }
                }
            }
            removedEntities.clear();
          
            // Ritar sprite objekt
            rootCanvas->drawSprites();
            SDL_RenderPresent(sys.getRen());
            capFrameRate(&renderTime, &remainder);

            if (saveImageCounter <= saveRenderedImageCap)
            {
                saveRenderedImage();
                saveImageCounter++;

            }

            if(loadNextLevel)
            {
                for(loadLevelFunc f : funcsOnLoadLevel)
                {
                    f();
                }
                loadNextLevel = false;
            }

            
        }
    }

    bool compareEntityLayer(Entity* e1, Entity* e2){
        return e1->getLayer() < e2->getLayer();
    }

    void Session::sortEntitiesByLayer(){
        std::sort(entities.begin(), entities.end(), compareEntityLayer);
    }

    void Session::clearEntities()
    {
        world->getCurrentLevel()->clearWallPointers();
        for (Entity* p : entities) { delete p; }
        entities.clear();
    }

    void Session::clearEntitiesExcept(std::string tag)
    {
        world->getCurrentLevel()->clearWallPointers();
        Entity* temp = findEntity(tag);
        for (Entity* p : entities) 
        { 
            if(p->tag != tag)
            {
                delete p;
            }
        }
        entities.clear();
        entities.push_back(temp);
    }

    Session::~Session()
    {
        clearEntities();
        funcsOnLoadLevel.clear();
        delete world;
        delete rootCanvas;
    }

    Session ses;

}


