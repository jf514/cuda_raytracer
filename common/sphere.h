#ifndef COMMON_SPHERE_H
#define COMMON_SPHERE_H
#pragma once

#include "hit.h"
#include "material.h"
#include "ray.h"
#include "vector.h"

#include <cmath>
#include <limits>


struct Sphere {

    Sphere() = default;

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
    Material* mat_ptr;
};

__host__ __device__ Hit collide(const Ray& ray, const Sphere& sphere, 
                                float tmin = 0, 
                                float tmax = std::numeric_limits<float>::max()){
    Vector3 oc = ray.o - sphere.center;
    float a = dot(ray.dir, ray.dir);
    float half_b = dot(oc, ray.dir);
    float c = dot(oc, oc) - sphere.radius*sphere.radius;

    float discriminant = half_b*half_b - a*c;
    if (discriminant < 0) return Hit::EmptyHit();
    float sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    float root = (-half_b - sqrtd) / a;
    if (root <= tmin || tmax <= root) {
        root = (-half_b + sqrtd) / a;
        if (root <= tmin || tmax <= root)
            return Hit::EmptyHit();
    }

    Hit hit;

    hit.hit = true;
    hit.t = root;
    //printf("root: %f\n", root);
    hit.position = ray.at(hit.t);
    hit.n = (hit.position - sphere.center) / sphere.radius;

    return hit;
}

#endif // COMMON_SPHERE