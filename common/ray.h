#ifndef COMMON_RAY_H
#define COMMON_RAY_H

#include "vector.h"

struct Ray {

    Ray() {}

    __host__ __device__ Ray(const Vector3& origin, const Vector3& dir) 
        : o(origin)
        , dir(dir)
        {}

    __host__ __device__ Vector3 at(float t) const { return o + t*dir; }

    // Origin
    Vector3 o;

    // Direction
    Vector3 dir;
}; 

#endif // COMMON_RAY_H