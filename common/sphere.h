#ifndef COMMON_SPHERE_H
#define COMMON_SPHERE_H
#pragma once

#include "ray.h"
#include "vector.h"

#include <cmath>
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

struct Sphere {
    __host__ __device__ Sphere(const Vector3& center, float radius) 
        : center(center)
        , radius(radius)
        {}

    Sphere(float x, float y, float z, float r) 
        : center(Vector3(x, y, z))
        , radius(r)
        {}

    Vector3 center;
    float radius;
};

__host__ __device__ Hit collide(const Ray& ray, const Sphere& sphere){
    Vector3 oc = ray.o - sphere.center;
    auto a = length_squared(ray.n);
    auto b = 2.0 * dot(oc, ray.n);
    auto c = dot(oc, oc) - sphere.radius*sphere.radius;
    auto discriminant = b*b - 4*a*c;

    if (discriminant < 0) {
        return Hit::EmptyHit();
    } else {
        Hit h;
        h.hit = true;
        // Prefer the closer hit (smaller root)
        h.t = (-b - std::sqrt(discriminant) ) / (2.0*a);
        // ...but if the ray starts inside, we may
        // have to consider the other root. 
        if(h.t < 0){
            h.t = (-b + std::sqrt(discriminant) ) / (2.0*a);
        }

        h.position = ray.at(h.t);
        Vector3 out_norm = normalize(h.position - sphere.center);

        // Set normal to always be opposite of ray, even 
        // the ray is coming from the inside.
        h.n = dot(out_norm, ray.n) < 0 ? out_norm : -out_norm;  
    
        return h;
    }
}

#endif // COMMON_SPHERE