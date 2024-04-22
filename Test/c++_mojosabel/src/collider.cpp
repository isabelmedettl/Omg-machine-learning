#include "Collider.h"

namespace mojosabel {

    Collider::Collider(int entX, int entY, int xOffset, int yOffset, int width, int height) : xOffset(xOffset), yOffset(yOffset)
    {
        rect.x = (entX + xOffset);
        rect.y = (entY + yOffset);
        rect.w = width;
        rect.h = height;
    }
    
    void Collider::shiftPosition(int xPos, int yPos)
    {
        rect.x = (xPos + xOffset);
        rect.y = (yPos + yOffset);
    }
}