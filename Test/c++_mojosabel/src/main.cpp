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
    int newEnemiesRequired = ses.getWorld()->getCurrentLevelIndex() + PROGRESSION_INCREMENT;
    generateGameObjects<GCrocodile>(ses.getWorld()->getCurrentLevel(), newEnemiesRequired, "images/Crocodile.png", true);
    ses.getWorld()->setMaxProgression(newEnemiesRequired);
    ses.getWorld()->currentProgressionValue = 0;
}

int main(int argc, char* argv[]) 
{

    std::cout << "***main***" << std::endl;

    UI = ses.getRootCanvas();

    //ses.createNewWorld(2, 48, 5, 4);
    ses.createNewWorld(2, 45, 5, 4);
    ses.getWorld()->newLevel("images/WaterTile.png", "images/WaterTileWithLilyPad.png");

    Vector2 spawnPos = ses.getWorld()->getCurrentLevel()->generateSpawnPosition();
    int spawnX = spawnPos.x;
    int spawnY = spawnPos.y;
    GTurtle* player = new GTurtle(spawnX, spawnY, 100);
    player->loadTexture(constants::gResPath + "images/Turtle.png");
    ses.add(player);

    generateGameObjects<GCrocodile>(ses.getWorld()->getCurrentLevel(), PROGRESSION_INCREMENT, "images/Crocodile.png", true );

    ses.addLoadLevelFunc(nextLevelFunc);
    ses.addLoadLevelFunc(enemiesToNextLevel);
    
    ses.run();

    std::cout << "***done***" << std::endl;

    return 0;
}

