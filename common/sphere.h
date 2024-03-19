#ifndef COMMON_SPHERE_H
#define COMMON_SPHERE_H
#pragma once

#include "hit.h"
#include "material.h"
#include "ray.h"
#include "vector.h"

#include <cmath>


struct Sphere {

    Sphere() = default;

    __HD__ Sphere(const Vector3& center, 
                                Real radius, Material* material) 
        : center(center)
        , radius(radius)
        , mat_ptr(material)
        {
            printf("Create sp.\n");
        }

    Sphere(Real x, Real y, Real z, Real r) 
        : center(Vector3(x, y, z))
        , radius(r)
        {
            printf("Create sp. 1\n");
        }

    __HD__ ~Sphere(){
            static int d = 0;
            printf("deleting sphere %i\n", d++);
            delete mat_ptr;
        }

    Vector3 center;
    Real radius;
    Material* mat_ptr;
};

__host__ __device__ inline Hit collide(const Ray& ray, const Sphere& sphere, 
                                Real tmin, Real tmax){
    Vector3 oc = ray.o - sphere.center;
    Real a = dot(ray.dir, ray.dir);
    Real half_b = dot(oc, ray.dir);
    Real c = dot(oc, oc) - sphere.radius*sphere.radius;

    Real discriminant = half_b*half_b - a*c;
    //printf("disc %f\n", discriminant);
    if (discriminant < 0) return Hit::EmptyHit();
    Real sqrtd = sqrt(discriminant);
    //printf("sqrtd %f\n", sqrtd);

    // Find the nearest root that lies in the acceptable range.
    Real root = (-half_b - sqrtd) / a;
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
    hit.material = sphere.mat_ptr;
    hit.n = (hit.position - sphere.center) / sphere.radius;

    return hit;
}

#endif // COMMON_SPHERE