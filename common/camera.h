#ifndef COMMON_CAMERA_H
#define COMMON_CAMERA_H
#pragma once

#include "ray.h"
#include "vector.h"
#include "printdb.h"

class Camera {
  public:

    Camera() {
        initialize();
    }

    double aspect_ratio      = 1.0;  // Ratio of image width over height
    int    image_width       = 512;  // Rendered image width in pixel count
    int    samples_per_pixel = 10;   // Count of random samples for each pixel
    int    max_depth         = 10;   // Maximum number of ray bounces into scene

    double vfov     = 90;              // Vertical view angle (field of view)
    Vector3 lookfrom = Vector3(0,0,-1);  // Point camera is looking from
    Vector3 lookat   = Vector3(0,0,0);   // Point camera is looking at
    Vector3   vup      = Vector3(0,1,0);     // Camera-relative "up" direction

    __host__ __device__ Ray get_ray(int i, int j) const;

  private:
    int    image_height;   // Rendered image height
    Vector3 center;         // Camera center
    Vector3 pixel00_loc;    // Location of pixel 0, 0
    Vector3   pixel_delta_u;  // Offset to pixel to the right
    Vector3   pixel_delta_v;  // Offset to pixel below
    Vector3   u, v, w;        // Camera frame basis vectors

    void initialize() {
        image_height = static_cast<int>(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        center = lookfrom;

        // Determine viewport dimensions.
        float focal_length = length(lookfrom - lookat);
        float theta = (3.1415926535/180.)*(vfov);
        float h = tan(theta/2);
        float viewport_height = 2 * h * focal_length;
        float viewport_width = viewport_height * (static_cast<double>(image_width)/image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = normalize(lookfrom - lookat);
        u = normalize(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        Vector3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
        Vector3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;
        printdb("du", pixel_delta_u);
        printdb("dv", pixel_delta_v);

        // Calculate the location of the upper left pixel.
        Vector3 viewport_upper_left = center - (focal_length * w) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }
};

__host__ __device__ Ray Camera::get_ray(int i, int j) const {
        Vector3 pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
        Vector3 pixel_sample = pixel_center;

        Vector3 ray_origin = center;
        Vector3 ray_direction = normalize(pixel_sample - ray_origin);
        Ray out(ray_origin, ray_direction);

        //printdb("out", out);

        return out;
}
#endif //COMMON_CAMERA_H