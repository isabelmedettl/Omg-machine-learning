#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <iostream>
#include <filesystem>
#include <sys/stat.h>
#include <windows.h>
#include "System.h"

namespace mojosabel {

    namespace fs = std::filesystem;

    System::System()
    {
        std::cout << "Hej det funkar: systemet!\n";
        SDL_Init(SDL_INIT_EVERYTHING);
        win = SDL_CreateWindow("Mojosabel", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, SCREEN_WIDTH, SCREEN_HEIGHT, 0);
        //SDL_SetWindowFullscreen(win, SDL_WINDOW_FULLSCREEN_DESKTOP);
        ren = SDL_CreateRenderer(win, -1, 0);
        TTF_Init();
        font = TTF_OpenFont((constants::gResPath + "fonts/arial.ttf").c_str(), 36);
        std::cout << "resPath: " << constants::gResPath << std::endl;
        //keyboard[MAX_KEYBOARD_KEYS] = {0};
        ensureFolderExists();

    }

    System::~System()
    {
        TTF_CloseFont(font);
        TTF_Quit();
        SDL_DestroyWindow(win);
        SDL_DestroyRenderer(ren);
        SDL_Quit();
    }

    SDL_Renderer* System::getRen() const 
    {
        return ren;
    }
    
    TTF_Font* System::getFont() const 
    {
        return font;
    }

    SDL_Window* System::getWin() const 
    {
        return win;
    }

    bool System::isOutOfBounds(int x, int y)
    {
        if (x > SCREEN_WIDTH || x < 0 || y > SCREEN_HEIGHT || y < 0)
        {
            return true;
        }
        return false;
    }

    int System::saveFolderExists_internal()
    {
        struct stat info;
        if (stat(constants::gSaveImagePath.c_str(), &info) != 0)
        {
            bSaveFolderExists = 0;
            return 0; // Cannot access
        }
        else if (info.st_mode & S_IFDIR)
        {
            bSaveFolderExists = 1;
            return 1; // It's a directory
        }  
        else
        {
            bSaveFolderExists = 0;
            return 0; // Not a directory
        }
    }

    int System::saveFolderExists()
    {
        return bSaveFolderExists;
    }

    void System::ensureFolderExists()
    {
        const char* pathToFolder = constants::gSaveImagePath.c_str();

        if (GetFileAttributesA(pathToFolder) == INVALID_FILE_ATTRIBUTES)
        {
            CreateDirectoryA(pathToFolder, NULL);
        }

        if (saveFolderExists_internal() == false)
        {
            SDL_Log("Folder does not exist, and wasn´t able to create one </333: %s", pathToFolder);
        }
    }

    void System::cleanDirectory()
    {
        cleanDirectory_internal();
    }

    void System::cleanDirectory_internal()
    {
        const char* pathToFolder = constants::gSaveImagePath.c_str();
        try 
        {
        // Check if the path exists and is a directory
            if (fs::exists(pathToFolder) && fs::is_directory(pathToFolder)) 
            {
                // Iterate through each item in the directory, check current entry and remove if found
                for (const auto& entry : fs::directory_iterator(pathToFolder)) 
                {
                    if (fs::is_regular_file(entry)) 
                    {
                        //fs::remove(entry.path());
                       std::cout << "remove everything in folder  " << constants::gSaveImagePath << std::endl;
                    }
                }
            }
        } 
        catch (const fs::filesystem_error& e) 
        {
            std::cerr << "Error cleaning directory: " << e.what() << '\n';
        }
    }
  
    System sys;    
}