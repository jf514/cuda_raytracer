#ifndef COMMON_PRINTDB_H
#define COMMON_PRINTDB_H
#pragma once

#include "ray.h"
#include "sphere.h"
#include "vector.h"

#include <stdio.h>
#include <iostream>

#ifndef __HD__ 
#define __HD__ __host__ __device__
#endif

__HD__ void printdb(const char* str){
    printf("%s\n", str);
}

__HD__ void printdb(const char* str, float val){
    printdb(str);
    printf("%f\n", val);
}

__HD__ void printdb(const char* str, const Vector3& v){
    printf("%s\n", str);
    printf("x: %f, y: %f, z: %f\n", v.x, v.y, v.z);
}

__HD__ void printdb(const char* str, const Ray& ray){
    printf("%s\n", str);
    printdb("ray.o", ray.o);
    printdb("ray.dir", ray.dir);
}

__HD__ void printdb(const char* str, const Hit& hit){
    printf("%s\n", str);
    printdb("hit pos:", hit.position);
    printdb("hit n:", hit.n);
    printdb("hit t", hit.t);
}

#endif // COMMON_PRINTDB_H