#ifndef COMMON_CAMERA_H
#define COMMON_CAMERA_H

#include "vector.h"

struct Camera {
    Vector3 pos;
    Vector3 dir;

    float width = 0;
    float aspect = 0;
    float f = 0;
};

#endif //COMMON_CAMERA_H