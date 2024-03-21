#ifndef COMMON_COMMON_H
#define COMMON_COMMON_H
#pragma once

#include <limits>

#define __HD__ __host__ __device__

typedef float Real;

constexpr Real max_flt = std::numeric_limits<Real>::max();

#endif // COMMON_COMMON_H