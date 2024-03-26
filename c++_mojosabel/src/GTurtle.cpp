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
    healthBar = Ui_label::getInstance((10), 0, 100, 50, "Health: " + std::to_string(getHealth()));
    ses.getRootCanvas()->addUiSprite(healthBar);
    isAlive = true;
}

void GTurtle::adjustHealth(int changeHealth){
    if (!isAlive) { return; }
    if((health + changeHealth) <= 0){
        health = 0;
        healthBar->setText("Health: " + std::to_string(getHealth()));
        isAlive = false;
        die();
        setCollision(false);
    } else {
        health += changeHealth;
        iFrameCounter = 0;
        healthBar->setText("Health: " + std::to_string(getHealth()));
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

class NewGameButton : public Ui_button
{
    private:
        bool isClickable = true;
        Canvas* canvas;
    public:
        NewGameButton(Canvas* c) : Ui_button((SCREEN_WIDTH/2 -100), 400, 200, 100, "New Game") { canvas = c; }
        void perform(Ui_button* source)
        {
            if (isClickable == true) {
                ses.clearEntitiesExcept("Player");
                delete ses.getWorld();
                ses.createNewWorld(2, 48, 5, 4);
                ses.getWorld()->newLevel("images/WaterTile.png", "images/WaterTileWithLilyPad.png");
                GTurtle* player = static_cast<GTurtle*>(ses.findEntity("Player"));
                generateGameObjects<GCrocodile>(ses.getWorld()->getCurrentLevel(), 5, "images/Crocodile.png", true );
                player->resetForNewLevel();
                isClickable = false;
                ses.getRootCanvas()->removeCanvas(canvas);
            }
        }
};

void GTurtle::die()
{
    Canvas* c = new Canvas(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
    c->addUiSprite(Ui_label::getInstance((SCREEN_WIDTH/2 -200), 200, 400, 80, "Oh no! You died"));
    Ui_button* uiButton = new NewGameButton(c);
    c->addUiSprite(uiButton);
    ses.getRootCanvas()->addCanvas(c);
    canShoot = false;
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
    healthBar->setText("Health: " + std::to_string(getHealth()));
    isAlive = true;
    canShoot = true;
}

