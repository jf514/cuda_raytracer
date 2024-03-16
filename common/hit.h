#ifndef COMMON_HIT_H
#define COMMON_HIT_H
#pragma once

#include "vector.h"

#include <limits>

struct Hit {
    __host__ __device__ Hit(){}
    __host__ __device__ Hit(const Hit& in){
        hit = in.hit;
        position = in.position;
        n = in.n;
    }

    // Did we collide?
    bool hit = false;
    // Location of hit in world
    Vector3 position;
    // Normal at hit point
    Vector3 n;

    float t = std::numeric_limits<float>::max();

    __host__ __device__ static Hit EmptyHit(){ return Hit(); }
};

#endif // COMMON_HIT_H
