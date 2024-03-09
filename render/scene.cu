#include "common.h"

#include <cstdio>
#include <fstream>
#include<iostream>


// GPU version of the same function
__global__ void Render(Vector3* img, int N){
    int tx = blockIdx.x*blockDim.x+threadIdx.x;
    int ty = blockIdx.y*blockDim.y+threadIdx.y;

    Sphere sphere(Vector3{float(N/2),float(N/2),10}, 120);
    Ray ray{Vector3{float(tx), float(ty), 0}, Vector3{0,0,1}};

    Hit h = collide(ray, sphere);
    if(h.hit){
        img[tx + N*ty] = 0.5*(1 + h.n);
    } else {
        img[tx + N*ty] = Vector3{0,0,0};
    }
}

// CPU version of the same function
void RenderCPU(Vector3* img, int N){
    Sphere sphere(Vector3{float(N/2),float(N/2),10}, 120);
    for(int tx = 0; tx < N; ++tx){
        for(int ty = 0; ty < N; ++ty){
            Ray ray{Vector3{float(tx), float(ty), 0}, Vector3{0,0,1}};

            Hit h = collide(ray, sphere);
            if(h.hit){
                img[tx + N*ty] = 0.5*(0.99 + h.n);
            } else {
                img[tx + N*ty] = Vector3{0,0,0};
            }
        }
    }
}

int main(int argc, char* argv[]){
    // Image side length - for this image size 
    // we expect CPU to be faster. For my architecture
    // I don't see the GPU going faster until 
    // N = 8 * 512, if we include memory transfer. However,
    // excluding transfer we see a speef up of a factor of
    // 100.
    const int N = 512;

    // Represent images as 1-D array of size N*N
    Vector3* img_h = new Vector3[N*N];

    // Render on GPU or CPU?
    float deltaT = 0;
    if(argc > 1 && std::string(argv[1]) == "-cpu"){
        std::cout << "Rendering on CPU...\n";
 
        Timer timer;
        tick(timer);
        RenderCPU(img_h, N);

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
        Render<<<numBlocks, blockThreadDist>>>(img_d, N);
        deltaT = tick(timer);

        cudaMemcpy(img_h, img_d, N*N*sizeof(Vector3), cudaMemcpyDeviceToHost);
        // Free memory
        cudaFree(img_d);
    }

    std::cout << "Finished. Took: " << 1000.*deltaT << " milliseconds.\n";

    WritePPM("test_out.ppm", img_h, N);
    delete[] img_h;

}