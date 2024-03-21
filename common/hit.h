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

    Real t = max_flt;
    Real u;
    Real v;
    bool front_face;

    __HD__ void set_face_normal(const Ray& r, const Vector3& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(r.dir, outward_normal) < 0;
        n = front_face ? outward_normal : -outward_normal;
    }

    __HD__ static Hit EmptyHit(){ return Hit(); }
};

#endif // COMMON_HIT_H
