#include <cmath>
#include "Vector2.h"


namespace mojosabel {

    Vector2 Vector2::moveTowards(Vector2 target, int sp)
    {
        float speed = sp;
        Vector2 delta = target - *this;
        if (magnitudeSquared() < speed * speed)
        {
            return target;
        }
        else 
        {
            return *this + delta.normalize() * speed;
        }
    }

    float Vector2::distance(Vector2 a, Vector2 b)
    {
        float sideA = abs(a.x - b.x);
        float sideB = abs(a.y - b.y);
        float distance = sqrt(pow(sideA, 2) + pow(sideB, 2));
        return distance;
    }

    float Vector2::magnitudeSquared()
    {
        return pow(x ,2) + pow(y, 2);
    }

    Vector2 Vector2::normalize()
    {
        if (x == 0 && y == 0) { return Vector2{0, 0}; }
        return *this / sqrt(magnitudeSquared());
    }

    Vector2 Vector2::operator+(const Vector2& rhs)
    {
        return Vector2{x + rhs.x, y + rhs.y};
    }

    Vector2 Vector2::operator*(float scale)
    {
        return Vector2{x*scale, y*scale};
    }

    Vector2 Vector2::operator-(const Vector2& rhs)
    {
        return Vector2{x - rhs.x, y - rhs.y};
    }

    Vector2 Vector2::operator/(float scale)
    {
        return Vector2{x/scale, y/scale};
    }
}