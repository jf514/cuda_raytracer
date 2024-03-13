#include "common.h"
#include "parallel.h"
#include "pcg.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <thread>

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

__host__ __device__ void RenderImpl(Vector3* img, Camera cam, Sphere* scene, int num_spheres, int tx, int ty){
    int N = cam.image_width;
    //tx = N/2;
    //ty = N/2;
    Ray ray = cam.get_ray(tx, ty);
    //printdb("Ray", ray);

    for(int i = 0; i < num_spheres; ++i){
        Hit h = collide(ray, scene[i]);
        if (h.hit) {
            //printdb("hit pos", h.position);
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

    RenderImpl(img, cam, scene, num_spheres, tx, ty);
}

void RenderCPU(Vector3* img, Camera cam, Sphere* scene, int num_spheres){
    int h = cam.image_width;
    int w = cam.image_width;

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    parallel_for([&](const Vector2i &tile) {
        // Use a different rng stream for each thread.
        pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, w);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, h);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                RenderImpl(img, cam, scene, num_spheres, x, y);
            }
        }
    }, Vector2i(num_tiles_x, num_tiles_y));
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
    cam.lookat = Vector3(0,0,1);
    cam.lookfrom = Vector3(0,0,0);
    cam.initialize();
    
    // Set up scene
    std::vector<Sphere> spheres;
    spheres.push_back(Sphere(Vector3{0,0,5}, 0.5));
    spheres.push_back(Sphere(Vector3{0,-100.5,-1}, 100));
    
    const int num_spheres = spheres.size();
    Sphere* scene_h = spheres.data();
 
    // Represent images as 1-D array of size N*N
    Vector3* img_h = new Vector3[N*N];

    // Render on GPU or CPU?
    float deltaT = 0;
    if(argc > 1 && std::string(argv[1]) == "-cpu"){
        std::cout << "Rendering on CPU...\n";
 
        //int num_threads = std::thread::hardware_concurrency();
        int num_threads = 50;
        parallel_init(num_threads);

        Timer timer;
        tick(timer);
        RenderCPU(img_h, cam, scene_h, num_spheres);

        deltaT = tick(timer);

        parallel_cleanup();

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
        cudaFree(scene_d);
    }

    std::cout << "Finished. Took: " << 1000.*deltaT << " milliseconds.\n";

    WritePPM("test_out.ppm", img_h, N);
    delete[] img_h;

    //delete that spheres

}