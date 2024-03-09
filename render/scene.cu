#include "utils.h"

#include <cstdio>
#include <fstream>
#include<iostream>


__global__ void Render(Vector3* img, int N){
    int tx = blockIdx.x*blockDim.x+threadIdx.x;
    int ty = blockIdx.y*blockDim.y+threadIdx.y;

    float x = tx - 0.5*N;
    float y = ty - 0.5*N;

    float rt = std::sqrt( x*x + y*y );
    if(rt < 0.25 * N) {
        img[tx + N*ty] = Vector3(0,0,0);
    } else {
        img[tx + N*ty] = Vector3(1,1,1);
    }
}

int main(void){

    // Image side length
    const long int N = 512;

    // Represent images as 1-D array of size N*N
    Vector3* img_h = new Vector3[N*N];

    // Device
    Vector3 *img_d;
    cudaMalloc(&img_d, N*N*sizeof(Vector3));

    const dim3 blockThreadDist(32, 32);
    const dim3 numBlocks(
        N/blockThreadDist.x,
        N/blockThreadDist.y
    );

    Render<<<numBlocks, blockThreadDist>>>(img_d, N);

    cudaMemcpy(img_h, img_d, N*N*sizeof(Vector3), cudaMemcpyDeviceToHost);

    WritePPM("test_out.ppm", img_h, N);

    // Free memory
    cudaFree(img_d);
    delete[] img_h;

    return 0;
}