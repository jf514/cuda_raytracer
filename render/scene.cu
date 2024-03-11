#include "common.h"

#include <cstdio>
#include <fstream>
#include<iostream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__host__ __device__ void RenderImp(Vector3* img, Camera cam, Sphere* scene, int num_spheres, int tx, int ty){
    int N = cam.image_width;
    Ray ray = cam.get_ray(tx, ty);
 
    for(int i = 0; i < num_spheres; ++i){
        Hit h = collide(ray, scene[i]);
        if (h.hit) {
            //printdb("hit");
            img[tx + N*ty] = 0.5f*Vector3(h.n.x+1.0f, h.n.y+1.0f, h.n.z+1.0f);
            return;
        }
    }

    float t = 0.5f*(ray.dir.y + 1.0f);
    img[tx + N*ty]=(1.0f-t)*Vector3(1.0, 1.0, 1.0) + t*Vector3(0.5, 0.7, 1.0);
}


// GPU version of the same function
__global__ void Render(Vector3* img, Camera cam, Sphere* scene, int num_spheres){
    int tx = blockIdx.x*blockDim.x+threadIdx.x;
    int ty = blockIdx.y*blockDim.y+threadIdx.y;

    RenderImp(img, cam, scene, num_spheres, tx, ty);
}

// CPU version of the same function
void RenderCPU(Vector3* img, Camera cam, Sphere* scene, int num_spheres){
    int N = cam.image_width;
    for(int tx = 0; tx < N; ++tx){
        for(int ty = 0; ty < N; ++ty){
            RenderImp(img, cam, scene, num_spheres, tx, ty);
        }
    }
}

int main(int argc, char* argv[]){
 
    // Image side length - for this image size 
    // we expect CPU to be faster. For my architecture
    // I don't see the GPU going faster until 
    // N = 8 * 512, if we include memory transfer. However,
    // excluding transfer we see a speed up of a factor of
    // 100.
    const int N = 512;

    // Set up camera
    Camera cam;
    cam.image_width = N;
    
    // Set up scene
    const int num_spheres = 2;
    Sphere* scene_h = new Sphere[num_spheres];
    scene_h[0] = Sphere(Vector3{0,0,-1}, 0.1);
    scene_h[1] = Sphere(Vector3{2,0,-3}, 2); 

    // Represent images as 1-D array of size N*N
    Vector3* img_h = new Vector3[N*N];

    // Render on GPU or CPU?
    float deltaT = 0;
    if(argc > 1 && std::string(argv[1]) == "-cpu"){
        std::cout << "Rendering on CPU...\n";
 
        Timer timer;
        tick(timer);
        RenderCPU(img_h, cam, scene_h, num_spheres);

        deltaT = tick(timer);

    } else {
        std::cout << "Rendering on GPU...\n";

        // Device
        Vector3 *img_d;
        cudaMalloc(&img_d, N*N*sizeof(Vector3));

        Sphere* scene_d;
        cudaMalloc(&scene_d, num_spheres*sizeof(Sphere));
        cudaMemcpy(scene_d,scene_h,num_spheres*sizeof(Sphere),cudaMemcpyHostToDevice);

        const dim3 blockThreadDist(32, 32);
        const dim3 numBlocks(
            N/blockThreadDist.x,
            N/blockThreadDist.y
        );

        Timer timer;
        tick(timer);
        Render<<<numBlocks, blockThreadDist>>>(img_d, cam, scene_d, num_spheres);
        deltaT = tick(timer);

        cudaMemcpy(img_h, img_d, N*N*sizeof(Vector3), cudaMemcpyDeviceToHost);
        // Free memory
        cudaFree(img_d);
    }

    std::cout << "Finished. Took: " << 1000.*deltaT << " milliseconds.\n";

    WritePPM("test_out.ppm", img_h, N);
    delete[] img_h;

}