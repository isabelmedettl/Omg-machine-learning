#include <iostream>
#include "Player.h"
#include "Bullet.h"
#include "System.h"
#include "Constants.h"
#include "Session.h"

namespace mojosabel {

    Player::Player(int x, int y, int w, int h, int layer, int sp)
        : GameObject (x, y, w, h, layer, "Player")
    {
        speed = sp;
        setCollision(true);
        std::cout << "player instantiated" << std::endl;
    }
    
    void Player::update()
    {
        move();
    }

    void Player::move()
    {
        if(sys.keyboard[KEY_W] && !(checkDirection("Up"))) { rect.y -= speed; }
        if(sys.keyboard[KEY_S] && !(checkDirection("Down"))) { rect.y += speed; }
        if(sys.keyboard[KEY_A] && !(checkDirection("Left"))) { rect.x -= speed; }
        if(sys.keyboard[KEY_D] && !(checkDirection("Right"))) { rect.x += speed; }
    }

    void Player::mouseDown(int x, int y)
    {
        fire(x, y);
    }

    void Player::fire(int x, int y)
    {
        Bullet *bullet = new Bullet(rect.x, rect.y, 0, 0, 0, "Bullet", 30, x, y);
        bullet->loadTexture(constants::gResPath + "images/Bullet.png");
        bullet->resizeToImage();
        bullet->setCollision(true);
        instantiate(bullet);
        hasColliders();
    }
}