#ifndef RAND_H
#define RAND_H
#pragma once

#include <curand.h>
#include <curand_kernel.h>

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

__HD__ inline vec3 random_unit_vector(void* rand_state) {
    return normalize(random_in_unit_sphere(rand_state));
}


__HD__ inline vec3 random_cosine_direction(void* rand_state) {
    auto r1 = NextRand(rand_state);
    auto r2 = NextRand(rand_state);

    auto phi = 2*pi*r1;
    auto x = cos(phi)*sqrt(r2);
    auto y = sin(phi)*sqrt(r2);
    auto z = sqrt(1-r2);

    return vec3(x, y, z);
}

#endif