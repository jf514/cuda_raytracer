#ifndef COMMON_MATERIAL_H
#define COMMON_MATERIAL_H
#pragma once 

#include "hit.h"
#include "pcg.h"
#include "pdf.h"
#include "ray.h"
#include "texture.h"

#include <curand_kernel.h>

class scatter_record {
  public:
    Vector3 attenuation;
    pdf* pdf_ptr;
    bool skip_pdf;
    Ray skip_pdf_Ray;
};

class Material {
  public:
    virtual ~Material() = default;

     __HD__ virtual Vector3 emitted(
        const Ray& r_in, const hit_record& rec, double u, double v, const point3& p
    ) const {
        return Vector3(0,0,0);
    }

     __HD__ virtual bool scatter(const Ray& r_in, const hit_record& rec, scatter_record& srec, void* rand_state) const {
        return false;
    }

     __HD__ virtual double scattering_pdf(const Ray& r_in, const hit_record& rec, const Ray& scattered)
    const {
        return 0;
    }
};

// JEF Todo FIX MEMORY LEAKS
class lambertian : public Material {
  public:
     __HD__ lambertian(const Vector3& a) : albedo(new solid_color(a)) {}
     __HD__ lambertian(texture* a) : albedo(a) {}

     __HD__ bool scatter(const Ray& r_in, const hit_record& rec, scatter_record& srec, void* rand_state) const override {
        srec.attenuation = albedo->value(rec.u, rec.v, rec.position);
        srec.pdf_ptr = new cosine_pdf(rec.n);
        srec.skip_pdf = false;
        return true;
    }

     __HD__ double scattering_pdf(const Ray& r_in, const hit_record& rec, const Ray& scattered) const override {
        auto cos_theta = dot(rec.n, normalize(scattered.dir));
        return cos_theta < 0 ? 0 : cos_theta/pi;
    }

  private:
    texture* albedo;
};


class metal : public Material {
  public:
     __HD__ metal(const Vector3& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

     __HD__ bool scatter(const Ray& r_in, const hit_record& rec, scatter_record& srec, void* rand_state) const override {
        srec.attenuation = albedo;
        srec.pdf_ptr = nullptr;
        srec.skip_pdf = true;
        vec3 reflected = reflect(normalize(r_in.dir), rec.n);
        srec.skip_pdf_Ray =
            Ray(rec.position, reflected + fuzz*random_in_unit_sphere(rand_state));
        return true;
    }

  private:
    Vector3 albedo;
    double fuzz;
};


class dielectric : public Material {
  public:
     __HD__ dielectric(double index_of_refraction) : ir(index_of_refraction) {}

     __HD__ bool scatter(const Ray& r_in, const hit_record& rec, scatter_record& srec, void* rand_state) const override {
        srec.attenuation = Vector3(1.0, 1.0, 1.0);
        srec.pdf_ptr = nullptr;
        srec.skip_pdf = true;
        double refraction_ratio = rec.front_face ? (1.0/ir) : ir;

        vec3 unit_direction = normalize(r_in.dir);
        double cos_theta = min(dot(-unit_direction, rec.n), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > NextRand(rand_state))
            direction = reflect(unit_direction, rec.n);
        else
            direction = refract(unit_direction, rec.n, refraction_ratio);

        srec.skip_pdf_Ray = Ray(rec.position, direction);
        return true;
    }

  private:
    double ir; // Index of Refraction

     __HD__ static double reflectance(double cosine, double ref_idx) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1-ref_idx) / (1+ref_idx);
        r0 = r0*r0;
        return r0 + (1-r0)*pow((1 - cosine),5);
    }
};


class diffuse_light : public Material {
  public:
     __HD__ diffuse_light(texture* a) : emit(a) {}
     __HD__ diffuse_light(Vector3 c) : emit(new solid_color(c)) {}

     __HD__ Vector3 emitted(const Ray& r_in, const hit_record& rec, double u, double v, const point3& p)
    const override {
        if (!rec.front_face)
            return Vector3(0,0,0);
        return emit->value(u, v, p);
    }

  private:
    texture* emit;
};


// class isotropic : public material {
//   public:
//     isotropic(Vector3 c) : albedo(make_shared<solid_Vector3>(c)) {}
//     isotropic(texture* a) : albedo(a) {}

//     bool scatter(const Ray& r_in, const hit_record& rec, scatter_record& srec, void* rand_state) const override {
//         srec.attenuation = albedo->value(rec.u, rec.v, rec.position);
//         srec.pdf_ptr = make_shared<sphere_pdf>();
//         srec.skip_pdf = false;
//         return true;
//     }

//     double scattering_pdf(const Ray& r_in, const hit_record& rec, const Ray& scattered)
//     const override {
//         return 1 / (4 * pi);
//     }

//   private:
//     texture* albedo;
// };


#endif