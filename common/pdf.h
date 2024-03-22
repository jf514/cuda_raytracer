#ifndef PDF_H
#define PDF_H
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

//#include "rtweekend.h"

//#include "hittable_list.h"
#include "onb.h"
#include "random.h"


class pdf {
  public:
    virtual ~pdf() {}

     __HD__ virtual double value(const vec3& direction) const = 0;
     __HD__ virtual vec3 generate(void* rand_state) const = 0;
};


class cosine_pdf : public pdf {
  public:
     __HD__ cosine_pdf(const vec3& w) { uvw.build_from_w(w); }

     __HD__ double value(const vec3& direction) const override {
        auto cosine_theta = dot(normalize(direction), uvw.w());
        return max(0., cosine_theta/pi);
    }

     __HD__ vec3 generate(void* rand_state) const override {
        return uvw.local(random_cosine_direction(rand_state));
    }

  private:
    onb uvw;
};


class sphere_pdf : public pdf {
  public:
     __HD__ sphere_pdf() { }

     __HD__ double value(const vec3& direction) const override {
        return 1/ (4 * pi);
    }

     __HD__ vec3 generate(void* rand_state) const override {
        return random_unit_vector(rand_state);
    }
};


// class hittable_pdf : public pdf {
//   public:
//     hittable_pdf(const hittable& _objects, const point3& _origin)
//       : objects(_objects), origin(_origin)
//     {}

//     double value(const vec3& direction) const override {
//         return objects.pdf_value(origin, direction);
//     }

//     vec3 generate(void* rand_state) const override {
//         return objects.random(origin);
//     }

//   private:
//     const hittable& objects;
//     point3 origin;
// };


// class mixture_pdf : public pdf {
//   public:
//     mixture_pdf(shared_ptr<pdf> p0, shared_ptr<pdf> p1) {
//         p[0] = p0;
//         p[1] = p1;
//     }

//     double value(const vec3& direction) const override {
//         return 0.5 * p[0]->value(direction) + 0.5 *p[1]->value(direction);
//     }

//     vec3 generate() const override {
//         if (random_double() < 0.5)
//             return p[0]->generate();
//         else
//             return p[1]->generate();
//     }

//   private:
//     shared_ptr<pdf> p[2];
// };


#endif
