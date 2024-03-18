#ifndef COMMON_HIT_H
#define COMMON_HIT_H
#pragma once

#include "common.h"
#include "vector.h"

// Fwd declare
class Material;

struct Hit {
    __host__ __device__ Hit(){}
    __host__ __device__ Hit(const Hit& in){
        hit = in.hit;
        t = in.t;
        position = in.position;
        n = in.n;
        material = in.material;
    }

    // Did we collide?
    bool hit = false;
    // Location of hit in world
    Vector3 position;
    // Normal at hit point
    Vector3 n;
    // Material type
    Material* material = nullptr;
    // Attenuation
    Vector3 attenuation;

    float t = max_flt;

    __host__ __device__ static Hit EmptyHit(){ return Hit(); }
};

#endif // COMMON_HIT_H
