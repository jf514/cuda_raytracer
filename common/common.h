#ifndef COMMON_COMMON_H
#define COMMON_COMMON_H
#pragma once

#include <limits>

#define __HD__ __host__ __device__

typedef float Real;

class Vector3;
typedef Vector3 vec3;
typedef Vector3 color;
typedef Vector3 point3;

struct Hit;
typedef Hit hit_record;

constexpr Real max_flt = std::numeric_limits<Real>::max();

// __HD__ Real fmax(a,b){ return ((a) > (b)) ? (a) : (b); } 

// __HD__ Real fmin(a,b){ return( ((a) < (b)) ? (a) : (b) ); }

#define pi 3.1415926535

#endif // COMMON_COMMON_H