#include <SDL2/SDL.h>
#include <string>
#include <iostream>
#include "Session.h"
#include "Ui_label.h"
#include "Ui_sprite.h"
#include "Ui_button.h"
#include "Canvas.h"
#include "Constants.h"
#include "Entity.h"
#include "MapGenerator.h"
#include "GameObjectGenerator.h"
#include "GCrocodile.h"
#include "GTurtle.h"


using namespace mojosabel;

int value = 0;
Canvas* UI;

void nextLevelFunc()
{
    ses.getWorld()->newLevel("images/WaterTile.png", "images/WaterTileWithLilyPad.png");
};

void enemiesToNextLevel()
{
    generateGameObjects<GCrocodile>(ses.getWorld()->getCurrentLevel(), (ses.getWorld()->getCurrentLevelIndex() * 4) + 5, "images/Crocodile.png", true);
}

int main(int argc, char* argv[]) 
{

    std::cout << "***main***" << std::endl;

    UI = ses.getRootCanvas();
    UI->addUiSprite(Ui_label::getInstance((SCREEN_WIDTH/2 -200), 0, 400, 100, "WOW SEA ADVENTURE 4000"));
    UI->addUiSprite(Ui_label::getInstance((SCREEN_WIDTH/2 -200), SCREEN_HEIGHT-100, 400, 100, "KILL ALL CROCODILES"));

    ses.createNewWorld(2, 48, 5, 4);
    ses.getWorld()->newLevel("images/WaterTile.png", "images/WaterTileWithLilyPad.png");

    Vector2 spawnPos = ses.getWorld()->getCurrentLevel()->generateSpawnPosition();
    int spawnX = spawnPos.x;
    int spawnY = spawnPos.y;
    GTurtle* player = new GTurtle(spawnX, spawnY, 100);
    player->loadTexture(constants::gResPath + "images/Turtle.png");
    ses.add(player);

    generateGameObjects<GCrocodile>(ses.getWorld()->getCurrentLevel(), 5, "images/Crocodile.png", true );

    ses.addLoadLevelFunc(nextLevelFunc);
    ses.addLoadLevelFunc(enemiesToNextLevel);
    
    ses.run();

    std::cout << "***done***" << std::endl;

    return 0;
}

