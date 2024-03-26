#include "GCrocodile.h"
#include "Session.h"
#include <iostream>

bool GCrocodile::checkPlayerDistance(){
    if(!getHasPlayer()){
        Entity* tempPlayer = ses.findEntity("Player");
        setPlayer((Player*) tempPlayer);
    }
    int difX = getPlayer()->position().x - rect.x;
    int difY = getPlayer()->position().y - rect.y;
    if(difX <= 128 && difX >= -128 && difY <= 128 && difY >= -128){
        return true;
    }
    return false;
}

void GCrocodile::update(){
    if (checkPlayerDistance()){
        moveToPlayer();
    } else {
        move();
    }
}

void GCrocodile::checkHealth()
{
    if (health <= 0)
    {
        destroy(this);
    }
}

void GCrocodile::moveToPlayer(){
    if(getPlayer()->position().x < rect.x && !checkDirection("Left")){
        mF = moveLeft;
        //std::cout << "Left" << std::endl;
    } else if(getPlayer()->position().x > rect.x && !checkDirection("Right")){
        mF = moveRight;
        //std::cout << "Right" << std::endl;
    } else if(getPlayer()->position().y < rect.y && !checkDirection("Up")){
        mF = moveUp;
        //std::cout << "Up" << std::endl;
    }else if(getPlayer()->position().y > rect.y && !checkDirection("Down")){
        mF = moveDown;
        //std::cout << "Down" << std::endl;
    }

    mF(this);
}