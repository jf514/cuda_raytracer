#ifndef COMMON_WORLD_H
#define COMMON_WORLD_H
#pragma once

#include "common.h"
#include "sphere.h"

struct World {

    __HD__ World(Sphere** spheres, int num_spheres) 
    : spheres(spheres)
    , num_spheres(num_spheres) 
    { 
        printf("const num_sph %i\n", num_spheres);
    }

    __HD__ ~World() {
        printf("deleting world, num spheres = %i\n", num_spheres);
        for(int i = 0; i < num_spheres; ++i){
            delete spheres[i];
        }
 
        //printf("start w/  delete[]\n");
        //delete[](spheres);
        //printf("done w/  delete[]\n");
    }

    __HD__ void Print(){
        printf("Num spheres = %i\n", num_spheres);
        for(int i = 0; i < num_spheres; ++i){
            printf("Sphere rad = %f\n", spheres[i]->radius);
        }
    }
    
    Sphere** spheres = nullptr;
    int num_spheres = 0;
};

// __HD__ World* CreateWorld(){
 
//     printf("Create world!\n");

//     int num_spheres = 2;
//     Sphere* spheres = new Sphere[num_spheres];

//     new(&spheres[0]) Sphere{
//                             Vector3{2,0,3},1.0,
//                             new Lambertian{Vector3{.8,.8,.3}}
//                         };

//     new(&spheres[1]) Sphere{
//                             Vector3{-2,0,3},1.0,
//                             new Metal{Vector3(.8,.8,.8), .3}
//                         };

//     printf("Create world!!\n");

//     World* w = new World(spheres, num_spheres);
//     printf("w = %p\n", w);

//     return w;

// }

__HD__ void CreateWorld(Sphere** spheres, World** world) {

    //printf("New CreateWorld\n");
    spheres[0] = new Sphere(Vector3(0,0,2), 0.5,
                            new Lambertian(Vector3(0.8, 0.3, 0.3)));
    spheres[1] = new Sphere(Vector3(0,-100.5,2), 100,
                            new Lambertian(Vector3(0.8, 0.8, 0.0)));
    spheres[2] = new Sphere(Vector3(1,0,2), 0.5,
                            new Metal(Vector3(0.8, 0.6, 0.2), 1.0));
    spheres[3] = new Sphere(Vector3(-1,0,2), 0.5,
                            new Metal(Vector3(0.8, 0.8, 0.8), 0.3));
    *world  = new World(spheres,4);
}


__HD__ Hit collideWorld(const Ray& ray, float t_min, float t_max, World* world){
    //printf("cw num_sph %i\n", world->num_spheres);
    // Closest hit
    Hit h_closest;
    for(int i = 0; i < world->num_spheres; ++i){
        Hit h = collide(ray, *world->spheres[i], 0, max_flt);
        // if(h.hit){
        //     printf("hittt! %f\n", h.t);
        // }
        if(h.hit && h.t < h_closest.t){
            //printf("hit!\n");
            h_closest = h;
        }
    }
    return h_closest;
}

#endif // COMMON_WORLD_H