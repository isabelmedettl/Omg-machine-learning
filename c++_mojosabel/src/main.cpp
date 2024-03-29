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
#include "JosPickup.h"
#include "JosPlayer.h"


using namespace mojosabel;

int value = 0;
Canvas* UI;

void nextLevelFunc()
{
    ses.getWorld()->newLevel("images/DesertTile32p.png", "images/DesertTileWithRock.png");
};

void pickupsForNextLevel()
{
    int newCrystalsRequired = ses.getWorld()->getCurrentLevelIndex() + 3;
    generateGameObjects<JosPickup>(ses.getWorld()->getCurrentLevel(), newCrystalsRequired, "images/Crystals32p.png", true);
    static_cast<JosPlayer*>(ses.findEntity("Player"))->setCrystalsRequired(newCrystalsRequired);
}

int main(int argc, char* argv[]) 
{

// add code to remove potential bitmaps 
    std::cout << "***main***" << std::endl;

    UI = ses.getRootCanvas();
    //UI->addUiSprite(Ui_label::getInstance((SCREEN_WIDTH/2 -300), 0, 600, 100, "OMG DESERT DIGGER ONE MILLION"));
    UI->addUiSprite(Ui_label::getInstance((SCREEN_WIDTH/2 -300), 0, 40, 40, "1"));

    UI->addUiSprite(Ui_label::getInstance((SCREEN_WIDTH/2 -200), SCREEN_HEIGHT-100, 400, 100, "MINE ALL MINERALS"));

    ses.createNewWorld(1, 50, 5, 4);
    ses.getWorld()->newLevel("images/DesertTile32p.png", "images/DesertTileWithRock.png");

    Vector2 spawnPos = ses.getWorld()->getCurrentLevel()->generateSpawnPosition();
    int spawnX = spawnPos.x;
    int spawnY = spawnPos.y;
    JosPlayer* player = new JosPlayer(spawnX, spawnY);
    ses.add(player);

    generateGameObjects<JosPickup>(ses.getWorld()->getCurrentLevel(), 3, "images/Crystals32p.png", true );
    player->setCrystalsRequired(3);

    ses.addLoadLevelFunc(nextLevelFunc);
    ses.addLoadLevelFunc(pickupsForNextLevel);
    
    ses.run();

    std::cout << "***done***" << std::endl;

    return 0;
}

