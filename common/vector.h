#ifndef COMMON_VECTOR3_H
#define COMMON_VECTOR3_H
#pragma once

#include <algorithm>
#include <math.h>
#include <cmath>
#include <ostream>

struct Vector2 {
    __host__ __device__ Vector2() {}

    __host__ __device__ Vector2(Real x, Real y) : x(x), y(y) {}
    
    __host__ __device__ Vector2(const Vector2 &v) : x(v.x), y(v.y) {}

    __host__ __device__ Real& operator[](int i) {
        return *(&x + i);
    }

    __host__ __device__ Real operator[](int i) const {
        return *(&x + i);
    }

    Real x, y;
};

struct Vector2i {
    Vector2i() {}

    Vector2i(int x, int y) : x(x), y(y) {}
    
    Vector2i(const Vector2i &v) : x(v.x), y(v.y) {}

    int& operator[](int i) {
        return *(&x + i);
    }

    int operator[](int i) const {
        return *(&x + i);
    }

    int x, y;
};

struct Vector3 {

    __host__ __device__ Vector3() { x = y = z = 0.; }

    __host__ __device__ Vector3(Real x, Real y, Real z) : x(x), y(y), z(z) {}
    
    __host__ __device__ Vector3(const Vector3 &v) : x(v.x), y(v.y), z(v.z) {}

    __host__ __device__ Real& operator[](int i) {
        return *(&x + i);
    }

    __host__ __device__ Real operator[](int i) const {
        return *(&x + i);
    }

    Real x, y, z;
};

struct Vector4 {
    __host__ __device__ Vector4() {}

    __host__ __device__ Vector4(Real x, Real y, Real z, Real w) : x(x), y(y), z(z), w(w) {}
    
    __host__ __device__ Vector4(const Vector4 &v) : x(v.x), y(v.y), z(v.z), w(v.w) {}


    __host__ __device__ Real& operator[](int i) {
        return *(&x + i);
    }

    __host__ __device__ Real operator[](int i) const {
        return *(&x + i);
    }

    Real x, y, z, w;
};

    __host__ __device__ inline Vector2 operator+(const Vector2 &v0, const Vector2 &v1) {
    return Vector2(v0.x + v1.x, v0.y + v1.y);
}

    __host__ __device__ inline Vector2 operator-(const Vector2 &v0, const Vector2 &v1) {
    return Vector2(v0.x - v1.x, v0.y - v1.y);
}

    __host__ __device__ inline Vector2 operator-(const Vector2 &v, Real s) {
    return Vector2(v.x - s, v.y - s);
}

    __host__ __device__ inline Vector2 operator-(Real s, const Vector2 &v) {
    return Vector2(s - v.x, s - v.y);
}

    __host__ __device__ inline Vector2 operator*(const Real &s, const Vector2 &v) {
    return Vector2(s * v[0], s * v[1]);
}

    __host__ __device__ inline Vector2 operator*(const Vector2 &v, const Real &s) {
    return Vector2(v[0] * s, v[1] * s);
}

    __host__ __device__ inline Vector2 operator/(const Vector2 &v, const Real &s) {
    return Vector2(v[0] / s, v[1] / s);
}


    __host__ __device__ inline Vector3 operator+(const Vector3 &v0, const Vector3 &v1) {
    return Vector3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
}

    __host__ __device__ inline Vector3 operator+(const Vector3 &v, const Real &s) {
    return Vector3(v.x + s, v.y + s, v.z + s);
}

    __host__ __device__ inline Vector3 operator+(const Real &s, const Vector3 &v) {
    return Vector3(s + v.x, s + v.y, s + v.z);
}

    __host__ __device__ inline Vector3& operator+=(Vector3 &v0, const Vector3 &v1) {
    v0.x += v1.x;
    v0.y += v1.y;
    v0.z += v1.z;
    return v0;
}

    __host__ __device__ inline Vector3 operator-(const Vector3 &v0, const Vector3 &v1) {
    return Vector3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}

    __host__ __device__ inline Vector3 operator-(Real s, const Vector3 &v) {
    return Vector3(s - v.x, s - v.y, s - v.z);
}

    __host__ __device__ inline Vector3 operator-(const Vector3 &v, Real s) {
    return Vector3(v.x - s, v.y - s, v.z - s);
}

    __host__ __device__ inline Vector3 operator-(const Vector3 &v) {
    return Vector3(-v.x, -v.y, -v.z);
}

    __host__ __device__ inline Vector3 operator*(const Real &s, const Vector3 &v) {
    return Vector3(s * v[0], s * v[1], s * v[2]);
}

    __host__ __device__ inline Vector3 operator*(const Vector3 &v, const Real &s) {
    return Vector3(v[0] * s, v[1] * s, v[2] * s);
}

    __host__ __device__ inline Vector3 operator*(const Vector3 &v0, const Vector3 &v1) {
    return Vector3(v0[0] * v1[0], v0[1] * v1[1], v0[2] * v1[2]);
}

    __host__ __device__ inline Vector3& operator*=(Vector3 &v, const Real &s) {
    v[0] *= s;
    v[1] *= s;
    v[2] *= s;
    return v;
}

    __host__ __device__ inline Vector3& operator*=(Vector3 &v0, const Vector3 &v1) {
    v0[0] *= v1[0];
    v0[1] *= v1[1];
    v0[2] *= v1[2];
    return v0;
}

    __host__ __device__ inline Vector3 operator/(const Vector3 &v, const Real &s) {
    Real inv_s = 1. / s;
    return Vector3(v[0] * inv_s, v[1] * inv_s, v[2] * inv_s);
}

    __host__ __device__ inline Vector3 operator/(const Real &s, const Vector3 &v) {
    return Vector3(s / v[0], s / v[1], s / v[2]);
}

    __host__ __device__ inline Vector3 operator/(const Vector3 &v0, const Vector3 &v1) {
    return Vector3(v0[0] / v1[0], v0[1] / v1[1], v0[2] / v1[2]);
}

    __host__ __device__ inline Vector3& operator/=(Vector3 &v, const Real &s) {
    Real inv_s = 1. / s;
    v *= inv_s;
    return v;
}

    __host__ __device__ inline Vector3 abs(const Vector3& v){
        return Vector3{fabsf(v.x), fabsf(v.y), fabsf(v.z)};
    }

    __host__ __device__ inline Real dot(const Vector3 &v0, const Vector3 &v1) {
    return v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
}

    __host__ __device__ inline Vector3 cross(const Vector3 &v0, const Vector3 &v1) {
    return Vector3{
        v0[1] * v1[2] - v0[2] * v1[1],
        v0[2] * v1[0] - v0[0] * v1[2],
        v0[0] * v1[1] - v0[1] * v1[0]};
}

    __host__ __device__ inline Real distance_squared(const Vector3 &v0, const Vector3 &v1) {
    return dot(v0 - v1, v0 - v1);
}

    __host__ __device__ inline Real distance(const Vector3 &v0, const Vector3 &v1) {
    return sqrt(distance_squared(v0, v1));
}

    __host__ __device__ inline Real length_squared(const Vector3 &v) {
    return dot(v, v);
}

    __host__ __device__ inline Real length(const Vector3 &v) {
    return sqrt(length_squared(v));
}

    __host__ __device__ inline Vector3 normalize(const Vector3 &v0) {
    auto l = length(v0);
    if (l <= 0) {
        return Vector3{0, 0, 0};
    } else {
        return v0 / l;
    }
}

    __host__ __device__ inline Real average(const Vector3 &v) {
    return (v.x + v.y + v.z) / 3;
}

//     __host__ __device__ inline Real max(const Vector3 &v) {
//     return std::max(std::max(v.x, v.y), v.z);
// }

//     __host__ __device__ inline Vector3 max(const Vector3 &v0, const Vector3 &v1) {
//     return Vector3{std::max(v0.x, v1.x), std::max(v0.y, v1.y), std::max(v0.z, v1.z)};
//}

//     __host__ __device__ inline bool isnan(const Vector2 &v) {
//     return isnan(v[0]) || isnan(v[1]);
// }

//     __host__ __device__ inline bool isnan(const Vector3 &v) {
//     return isnan(v[0]) || isnan(v[1]) || isnan(v[2]);
// }

//     __host__ __device__ inline bool isfinite(const Vector2 &v) {
//     return isfinite(v[0]) || isfinite(v[1]);
// }

//     __host__ __device__ inline bool isfinite(const Vector3 &v) {
//     return isfinite(v[0]) || isfinite(v[1]) || isfinite(v[2]);
// }

//     __host__ __device__ inline std::ostream& operator<<(std::ostream &os, const Vector2 &v) {
//     return os << "(" << v[0] << ", " << v[1] << ")";
// }

//     __host__ __device__ inline std::ostream& operator<<(std::ostream &os, const Vector3 &v) {
//     return os << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";
// }

#endif // COMMON_VECTOR3_H