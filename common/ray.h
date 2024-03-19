#ifndef COMMON_RAY_H
#define COMMON_RAY_H

#include "common.h"
#include "vector.h"

struct Ray {

    __HD__ Ray() {}

    __HD__ Ray(const Vector3& origin, const Vector3& dir) 
        : o(origin)
        , dir(dir)
        {}

    __host__ __device__ Vector3 at(Real t) const { return o + t*dir; }

    // Origin
    Vector3 o;

    // Direction
    Vector3 dir;
}; 

#endif // COMMON_RAY_H