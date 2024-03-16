#ifndef COMMON_MATERIAL_H
#define COMMON_MATERIAL_H
#pragma once 

#include "ray.h"
#include "sphere.h"

#include <curand_kernel.h>

#define RANDVECTOR3 Vector3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ Vector3 random_in_unit_sphere(curandState *local_rand_state) {
    Vector3 p;
    do {
        p = 2.0f*RANDVECTOR3 - Vector3(1,1,1);
    } while (dot(p,p) >= 1.0f);
    return p;
}

__device__ Vector3 reflect(const Vector3& v, const Vector3& n) {
     return v - 2.0f*dot(v,n)*n;
}

class Material  {
    public:
        __device__ virtual bool scatter(const Ray& r_in, const Hit& rec, Vector3& attenuation, Ray& scattered, curandState *local_rand_state) const = 0;
};

class Lambertian : public Material {
    public:
        __device__ Lambertian(const Vector3& a) : albedo(a) {}

        __device__ virtual bool scatter(const Ray& r_in, const Hit& rec, Vector3& attenuation, Ray& scattered, curandState *local_rand_state) const  {
             Vector3 target = rec.position + rec.n + random_in_unit_sphere(local_rand_state);
             scattered = Ray(rec.position, target-rec.position);
             attenuation = albedo;
             return true;
        }

        Vector3 albedo;
};

class Metal : public Material {
    public:
        __device__ Metal(const Vector3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
        
        __device__ virtual bool scatter(const Ray& r_in, const Hit& rec, Vector3& attenuation, Ray& scattered, curandState *local_rand_state) const  {
            Vector3 reflected = reflect(normalize(r_in.dir), rec.n);
            scattered = Ray(rec.position, reflected + fuzz*random_in_unit_sphere(local_rand_state));
            attenuation = albedo;
            return (dot(scattered.dir, rec.n) > 0.0f);
        }
        Vector3 albedo;
        float fuzz;
};

#endif