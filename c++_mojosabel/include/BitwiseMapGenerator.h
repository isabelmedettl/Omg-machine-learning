#ifndef BITWISEMAPGENERATOR_H
#define BITWISEMAPGENERATOR_H

#include "System.h"
#include "Level.h"

namespace mojosabel {
    class BitwiseMapGenerator {
    private:
        int smc, rfp, swl, sul;
        unsigned long int bitMap[MAP_HEIGHT]{0UL};
        int getSurroundingWallCount(int tileX, int tileY);
        void randomFillBitMap();
        void smoothMap();
    public:
        BitwiseMapGenerator() = delete;
        BitwiseMapGenerator(const BitwiseMapGenerator*) = delete;
        const BitwiseMapGenerator operator=(const BitwiseMapGenerator&) = delete;
        BitwiseMapGenerator(int smoothMap, int fillPercent, int smoothWalkableLimit, int smoothUnwalkableLimit)
            : smc(smoothMap), rfp(fillPercent), swl(smoothWalkableLimit), sul(smoothUnwalkableLimit){}
        void generateBitMap();
        void mixMap(Level* level);
    };
}

#endif