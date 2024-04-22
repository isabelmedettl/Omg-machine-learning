#ifndef COLLISION_H
#define COLLISION_H

namespace mojosabel {

    template <typename T>
    class Collision
    {
    public:
        T* object;
        const std::string tag;
        Collision(T* object, std::string tag) : object(object), tag(tag) {}
        Collision() = delete;
    };
}

#endif