#include "GTurtle.h"
#include <iostream>
#include "Session.h"
#include "Ui_label.h"
#include "Ui_sprite.h"
#include "Ui_button.h"
#include "GShellBullet.h"
#include "GameObjectGenerator.h"
#include "GCrocodile.h"


GTurtle::GTurtle(int x, int y, int startHealth) : Player(x, y, 32, 32, 0, 3)
{
    health = startHealth; canShoot = true;
    isAlive = true;
}

void GTurtle::adjustHealth(int changeHealth){
    if (!isAlive) { return; }
    if((health + changeHealth) <= 0){
        health = 0;
        isAlive = false;
        die();
        setCollision(false);
    } else {
        health += changeHealth;
        iFrameCounter = 0;
        setCollision(false);
    }
}

void GTurtle::onCollision(Collision<Entity> collision){
    if(collision.tag == "Enemy"){
        adjustHealth(-10);
    }
}

void GTurtle::update(){
    move();

    shootCounter++;
    if(shootCounter >= shootCoolDown && sys.keyboard[KEY_SPACE] == 1)
    {
        if ((sys.keyboard[KEY_D] + sys.keyboard[KEY_A] + sys.keyboard[KEY_S] + sys.keyboard[KEY_W]) == 0)
        {
            fire(rect.x, rect.y + 13);
        }
        else 
        {
            fire(rect.x + (sys.keyboard[KEY_D] - sys.keyboard[KEY_A]) * 13, rect.y + (sys.keyboard[KEY_S] - sys.keyboard[KEY_W]) * 13);
        }
        shootCounter = 0;
    }
    
    iFrameCounter++;
    if(iFrameCounter > 2000000000){
        iFrameCounter = 60;
    }
    if(iFrameCounter > 60 && health > 0){
        setCollision(true);
    }
    
}

bool GTurtle::levelCompleted(){
    Entity* croc = ses.findEntity("Enemy");
    if(croc == nullptr){
        return true;
    } else {
        return false;
    }
}


void GTurtle::die()
{
    ses.bRenderBlackScreen = true;
}

void GTurtle::fire(int x, int y)
{
    if (canShoot) 
    {
        GShellBullet *bullet = new GShellBullet(rect.x, rect.y, x, y);
        bullet->loadTexture(constants::gResPath + "images/Bullet.png");
        bullet->setCollision(true);
        instantiate(bullet);
        hasColliders();
    }
    
}

void GTurtle::resetForNewLevel() 
{
    setHealth(100);
    isAlive = true;
    canShoot = true;
}

