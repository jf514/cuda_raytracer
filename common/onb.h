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

#include "common.h"
#include "onb.h"


class pdf {
  public:
    virtual ~pdf() {}

    virtual Real value(const Vector3& direction) const = 0;
    virtual Vector3 generate() const = 0;
};


class cosine_pdf : public pdf {
  public:
    cosine_pdf(const Vector3& w) { uvw.build_from_w(w); }

    Real value(const Vector3& direction) const override {
        auto cosine_theta = dot(normalize(direction), uvw.w());
        return fmax(0, cosine_theta/pi);
    }

    Vector3 generate() const override {
        return uvw.local(random_cosine_direction());
    }

  private:
    onb uvw;
};


class sphere_pdf : public pdf {
  public:
    sphere_pdf() { }

    Real value(const Vector3& direction) const override {
        return 1/ (4 * pi);
    }

    Vector3 generate() const override {
        return random_unit_vector();
    }
};


class hittable_pdf : public pdf {
  public:
    hittable_pdf(const hittable& _objects, const Vector3& _origin)
      : objects(_objects), origin(_origin)
    {}

    Real value(const Vector3& direction) const override {
        return objects.pdf_value(origin, direction);
    }

    Vector3 generate() const override {
        return objects.random(origin);
    }

  private:
    const hittable& objects;
    Vector3 origin;
};


class mixture_pdf : public pdf {
  public:
    mixture_pdf(pdf* p0, pdf* p1) {
        p[0] = p0;
        p[1] = p1;
    }

    Real value(const Vector3& direction) const override {
        return 0.5 * p[0]->value(direction) + 0.5 *p[1]->value(direction);
    }

    Vector3 generate() const override {
        if (random_Real() < 0.5)
            return p[0]->generate();
        else
            return p[1]->generate();
    }

  private:
    pdf* p[2];
};


#endif
