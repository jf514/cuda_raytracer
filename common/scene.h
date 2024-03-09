#ifndef COMMON_SCENE_H
#define COMMON_SCENE_H

#include "camera.h"
#include "sphere.h"

#include <vector>

struct Scene {
    Camera cam;
    std::vector<Sphere> spheres;
};


#endif  // COMMON_SCENE_H

