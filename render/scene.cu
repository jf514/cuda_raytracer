#include "common.h"

#include <cstdio>
#include <fstream>
#include<iostream>

__host__ __device__ void RenderImp(Vector3* img, Camera cam, int tx, int ty){
    int N = cam.image_width;
    Sphere sphere( Vector3{0,0,1}, 1);
    Ray ray = cam.get_ray(tx, ty);

    Hit h = collide(ray, sphere);
    if(h.hit){
        img[tx + N*ty] = 0.5*(1 + h.n);
    } else {
        img[tx + N*ty] = Vector3{0,0,0};
    }
}


// GPU version of the same function
__global__ void Render(Vector3* img, Camera cam){
    int tx = blockIdx.x*blockDim.x+threadIdx.x;
    int ty = blockIdx.y*blockDim.y+threadIdx.y;

    RenderImp(img, cam, tx, ty);
}

// CPU version of the same function
void RenderCPU(Vector3* img, Camera cam){
    int N = cam.image_width;
    for(int tx = 0; tx < N; ++tx){
        for(int ty = 0; ty < N; ++ty){
            RenderImp(img, cam, tx, ty);
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
    Sphere* scene = new Sphere[2];
    scene[0] = Sphere(Vector3{0,0,5}, 1);
    scene[1] = Sphere(Vector3{2,2,5}, 2); 

    // Represent images as 1-D array of size N*N
    Vector3* img_h = new Vector3[N*N];

    // Render on GPU or CPU?
    float deltaT = 0;
    if(argc > 1 && std::string(argv[1]) == "-cpu"){
        std::cout << "Rendering on CPU...\n";
 
        Timer timer;
        tick(timer);
        RenderCPU(img_h, cam);

        deltaT = tick(timer);

    } else {
        std::cout << "Rendering on GPU...\n";

        // Device
        Vector3 *img_d;
        cudaMalloc(&img_d, N*N*sizeof(Vector3));

        const dim3 blockThreadDist(32, 32);
        const dim3 numBlocks(
            N/blockThreadDist.x,
            N/blockThreadDist.y
        );

        Timer timer;
        tick(timer);
        Render<<<numBlocks, blockThreadDist>>>(img_d, cam);
        deltaT = tick(timer);

        cudaMemcpy(img_h, img_d, N*N*sizeof(Vector3), cudaMemcpyDeviceToHost);
        // Free memory
        cudaFree(img_d);
    }

    std::cout << "Finished. Took: " << 1000.*deltaT << " milliseconds.\n";

    WritePPM("test_out.ppm", img_h, N);
    delete[] img_h;

}