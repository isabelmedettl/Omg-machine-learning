#ifndef SESSION_H
#define SESSION_H

#include <vector>

#include "World.h"
#include "Canvas.h"
#include "Entity.h"


class System;
namespace mojosabel {

    class Session
    {
    private:
        World* world;
        long renderTime; //tiden det tog för förra framen att rendera
        float remainder; 
        SDL_Window* window = nullptr;
        bool loadNextLevel = false;
        Canvas *rootCanvas;
        std::vector<Entity*> entities, addedEntities, removedEntities;
        typedef void (*loadLevelFunc)();
        std::vector<loadLevelFunc> funcsOnLoadLevel;
        void doKeyUp(SDL_KeyboardEvent* event);
        void doKeyDown(SDL_KeyboardEvent* event);
        bool entityExists(Entity* entity); //loopa genom alla entities, om en entity är samma returnera true, annars returnera false efter loopen
        bool checkCollision(SDL_Rect rect1, SDL_Rect rect2);
        bool checkColliders(SDL_Rect rectToCheck, std::vector<Collider>& colliders);
        void checkAllCollisions(Entity* entity); // går igenom alla entities och kollar om det objektet kolliderar med något av objekten 
        void capFrameRate(long *renderTime, float *remainder);
        void sortEntitiesByLayer();

        int currentLevel = 0;
        int currentProgressionMaxValue = 0;
        int currentProgressionValue = 0;

        SDL_Surface* cachedSurface = nullptr;

    public:
        Session();
        Session(const Session* other) = delete;
        const Session operator=(const Session& rhs) = delete;
        void add(Entity* entityToAdd);
        void remove(Entity* entityToRemove);
        void createNewWorld(int smoothMap, int fillPercent, int smoothWalkableLimit, int smoothUnwalkableLimit);
        void run();
        void clearEntities();
        void clearEntitiesExcept(std::string tag);
        void setLoadNextLevel(bool toSet) { loadNextLevel = toSet; }
        void addLoadLevelFunc(loadLevelFunc funcToAdd) 
        { 
            funcsOnLoadLevel.push_back(funcToAdd); 
            updateCurrentProgressionInfo();
        }

        void updateCurrentPickupCount();
        void updateCurrentProgressionInfo();
        World* getWorld() { return world; }
        ~Session();
        Canvas* getRootCanvas() {return rootCanvas;};
        Entity* findEntity(std::string name);
        void renderSliders(SDL_Renderer* renderer);

    private:
         /* === Saving and creating files */
        //int compareToCachedSurface(SDL_Surface* const surface);
        //void saveRenderedImage();
    };

    // struct collision som innehåller en pointer/namnet till det objektet kolliderar med
    extern Session ses;
    
}

#endif