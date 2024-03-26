#ifndef VECTOR2_H
#define VECTOR2_H

#include <SDL2/SDL.h>

namespace mojosabel {

    struct Vector2
    {
        float x;
        float y;
        Vector2() : Vector2(0, 0) {}
        Vector2(float x, float y) : x(x), y(y) {}
        Vector2(SDL_Rect rect) : x(rect.x), y(rect.y) {}
        Vector2(int x, int y) : x(x), y(y) {}
        Vector2 moveTowards(Vector2 target, int speed);
        float distance(Vector2 a, Vector2 b);

        Vector2 operator+(const Vector2& rhs);
        Vector2 operator*(float scale);
        Vector2 operator-(const Vector2& rhs);
        Vector2 operator/(float scale);
        float magnitudeSquared();
        Vector2 normalize();
    };
}

#endif