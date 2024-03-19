#ifndef COMMON_MATERIAL_H
#define COMMON_MATERIAL_H
#pragma once 

#include "hit.h"
#include "pcg.h"
#include "ray.h"

#include <curand_kernel.h>

#define RANDVECTOR3_GPU Vector3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__HD__ Vector3 GetRandVec3(void* rand_state){
    #ifdef __CUDA_ARCH__
        auto local_rand_state = reinterpret_cast<curandState*>(rand_state);
        return Vector3(
                curand_uniform(local_rand_state),
                curand_uniform(local_rand_state),
                curand_uniform(local_rand_state));
    #else
        auto rng = reinterpret_cast<pcg32_state*>(rand_state);
        return Vector3(
                    next_pcg32_real<Real>(*rng),
                    next_pcg32_real<Real>(*rng),
                    next_pcg32_real<Real>(*rng));
    #endif
}


__HD__ Vector3 random_in_unit_sphere(void* rand_state) {
    Vector3 p;
    do {
        p = 2.0f*GetRandVec3(rand_state) - Vector3(1,1,1);
    } while (dot(p,p) >= 1.0f);
    return p;
}

__HD__ Vector3 reflect(const Vector3& v, const Vector3& n) {
     return v - 2.0f*dot(v,n)*n;
}

class Material  {
    public:
        __HD__ virtual bool scatter(const Ray& r_in, const Hit& rec, 
            Vector3& attenuation, Ray& scattered, void* rand_state) const = 0;
};

class Lambertian : public Material {
    public:
        __HD__ Lambertian(const Vector3& a) : albedo(a) {}

        __HD__ virtual bool scatter(const Ray& r_in, const Hit& rec, 
                                    Vector3& attenuation, Ray& scattered, 
                                    void* rand_state) const  {
             Vector3 target = rec.position + rec.n + random_in_unit_sphere(rand_state);
             scattered = Ray(rec.position, target-rec.position);
             attenuation = albedo;
             return true;
        }

        Vector3 albedo;
};

class Metal : public Material {
    public:
        __HD__ Metal(const Vector3& a, Real f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
        
        __HD__ virtual bool scatter(const Ray& r_in, const Hit& rec, 
                                    Vector3& attenuation, Ray& scattered, 
                                    void* rand_state) const  {
            Vector3 reflected = reflect(normalize(r_in.dir), rec.n);
            scattered = Ray(rec.position, reflected + fuzz*random_in_unit_sphere(rand_state));
            attenuation = albedo;
            return (dot(scattered.dir, rec.n) > 0.0f);
        }

        Vector3 albedo;
        Real fuzz;
};

#endif