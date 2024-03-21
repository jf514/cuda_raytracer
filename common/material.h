#ifndef COMMON_MATERIAL_H
#define COMMON_MATERIAL_H
#pragma once 

#include "hit.h"
#include "pcg.h"
#include "ray.h"

#include <curand_kernel.h>

class scatter_record {
  public:
    Vector3 attenuation;
    pdf* pdf_ptr;
    bool skip_pdf;
    Ray skip_pdf_ray;
};


__HD__ Real schlick(Real cosine, Real ref_idx) {
    Real r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*pow((1.0f - cosine),5.0f);
}

__HD__ bool refract(const Vector3& v, const Vector3& n, Real ni_over_nt, Vector3& refracted) {
    Vector3 uv = normalize(v);
    Real dt = dot(uv, n);
    Real discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > 0) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
        return false;
}

__HD__ inline Real NextRand(void* rand_state){
#ifdef __CUDA_ARCH__
    auto local_rand_state = reinterpret_cast<curandState*>(rand_state);
    return curand_uniform(local_rand_state);
#else
    auto rng = reinterpret_cast<pcg32_state*>(rand_state);
    return next_pcg32_real<Real>(*rng);
#endif
}

__HD__ inline Vector3 GetRandVec3(void* rand_state){
    return Vector3(NextRand(rand_state),
            NextRand(rand_state),
            NextRand(rand_state));
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

        __HD__ virtual Vector3 emit(
            const Ray& r_in, const Hit& rec) const {
            return Vector3(0,0,0);
        }

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

class Dielectric : public Material {
public:
    __HD__ Dielectric(Real ri) : ref_idx(ri) {}
    __HD__ virtual bool scatter(const Ray& r_in,
                         const Hit& rec,
                         Vector3& attenuation,
                         Ray& scattered,
                         void* rand_state) const  {
        Vector3 outward_normal;
        Vector3 reflected = reflect(r_in.dir, rec.n);
        Real ni_over_nt;
        attenuation = Vector3(1.0, 1.0, 1.0);
        Vector3 refracted;
        Real reflect_prob;
        Real cosine;
        if (dot(r_in.dir, rec.n) > 0.0f) {
            outward_normal = -rec.n;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.dir, rec.n) / length(r_in.dir);
            cosine = sqrt(1.0f - ref_idx*ref_idx*(1-cosine*cosine));
        }
        else {
            outward_normal = rec.n;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.dir, rec.n) / length(r_in.dir);
        }
        if (refract(r_in.dir, outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0f;
        if (NextRand(rand_state) < reflect_prob)
            scattered = Ray(rec.position, reflected);
        else
            scattered = Ray(rec.position, refracted);
        return true;
    }

    Real ref_idx;
};

class DiffuseLight : public Material {
  public:
    __HD__ DiffuseLight(Vector3 c) : color(c) {}

    __HD__ Vector3 emit(const Ray& r_in, const Hit& rec)
    const override {
        // Hitting front face
        if (dot(rec.n, r_in.dir) > 0)
            return Vector3(0,0,0);
        return color;
    }

  private:
    Vector3 color;
};

#endif