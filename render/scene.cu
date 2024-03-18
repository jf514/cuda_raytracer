#include "camera.h"
#include "common.h"
#include "hit.h"
#include "parallel.h"
#include "pcg.h"
#include "ray.h"
#include "sphere.h"
#include "timer.h"
#include "utils.h"
#include "world.h"

#include <cstdio>
#include <curand_kernel.h>
#include <curand.h>
#include <fstream>
#include <iostream>
#include <thread>

#define CHECK_CUDA_ERRORS(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        std::cerr << cudaGetErrorString(result) << "\n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__HD__ Vector3 Color(const Ray& r, World* world, void* rand_state) {
    Ray cur_ray = r;
    Vector3 cur_attenuation = Vector3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        Hit hit = collideWorld(cur_ray, 0.001f, max_flt, world);
        if (hit.hit) {
            Ray scattered;
            Vector3 attenuation;
            if(hit.material->scatter(cur_ray, hit, attenuation, scattered, rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return Vector3(0.0,0.0,0.0);
            }
        }
        else {
            float t = 0.5f*(cur_ray.dir.y + 1.0f);
            Vector3 c = (1.0f-t)*Vector3(1.0, 1.0, 1.0) + t*Vector3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return Vector3(0.0,0.0,0.0); // exceeded recursion
}

__host__ __device__ Vector3 RenderImpl(Vector3* img, Camera cam, 
        Sphere* scene, int num_spheres, Ray ray){
    for(int i = 0; i < num_spheres; ++i){
        Hit h = collide(ray, scene[i], 0, max_flt);
        if (h.hit) {
            //printdb("hit pos", h.position);
            return 0.5f*Vector3(h.n.x+1.0f, h.n.y+1.0f, h.n.z+1.0f);
        }
    }

    float t = 0.5f*(ray.dir.y + 1.0f);
    return (1.0f-t)*Vector3(1.0, 1.0, 1.0) + t*Vector3(0.5, 0.7, 1.0);
}

__global__ void RenderInit(int max_x, int max_y, World** world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i == 0) && (j == 0)){
        *world = CreateWorld();
    }

    if((i >= max_x) || (j >= max_y)) 
        return;
    
    int pixel_index = j*max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void RenderCleanup(World** world){
    delete *world;
}

// GPU version of the same function
__global__ void Render(Vector3* img, Camera cam, World** world_d, 
                        curandState* rand_state){
    int tx = blockIdx.x*blockDim.x+threadIdx.x;
    int ty = blockIdx.y*blockDim.y+threadIdx.y;

    if((tx >= cam.image_width) || (ty >= cam.get_height())){ 
        return;
    }

    Vector3 color(0,0,0);
    for(int s = 0; s < cam.samples_per_pixel; ++s){
        int pixel_index = ty*cam.image_width + tx;
        curandState* local_rand_state = &rand_state[pixel_index];
        float rnd_x = curand_uniform(local_rand_state);
        float rnd_y = curand_uniform(local_rand_state);
        Ray ray = cam.get_ray(tx, ty, rnd_x, rnd_y);
        color += Color(ray, *world_d, local_rand_state);
    }

    color = color/float(cam.samples_per_pixel);
    img[tx + cam.image_width*ty] = color;
}

void RenderCPU(Vector3* img, Camera cam, World* world){
    int h = cam.image_width;
    int w = cam.image_width;

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    //curandState cstate;
    //curand_init(1984, 0, 0, &cstate);

    parallel_for([&](const Vector2i &tile) {
        // Use a different rng stream for each thread.
        pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, w);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, h);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                Vector3 color(0,0,0);
                for(int s = 0; s < cam.samples_per_pixel; ++s){
                    float rnd_x = next_pcg32_real<float>(rng);
                    float rnd_y = next_pcg32_real<float>(rng);
                    Ray ray = cam.get_ray(x, y, rnd_x, rnd_y);
                    color += Color(ray, world, &rng);
                }
                color /= cam.samples_per_pixel;
                img[x + w*y] = color;
            }
        }
    }, Vector2i(num_tiles_x, num_tiles_y));
}

// __global__ void WorldCreate(World** world){
//     *world = CreateWorld();
// }

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
       
    // Represent images as 1-D array of size N*N
    Vector3* img_h = new Vector3[N*N];

    // Render on GPU or CPU?
    float deltaT = 0;
    if(argc > 1 && std::string(argv[1]) == "-cpu"){
        std::cout << "Rendering on CPU...\n";

        World* w_h = CreateWorld();
 
        //int num_threads = std::thread::hardware_concurrency();
        int num_threads = 50;
        parallel_init(num_threads);

        Timer timer;
        tick(timer);
        RenderCPU(img_h, cam, w_h);
        deltaT = tick(timer);

        parallel_cleanup();
    } else {
        std::cout << "Rendering on GPU...\n";

        // Allocate device memory.
        Vector3 *img_d;
        CHECK_CUDA_ERRORS(cudaMalloc(&img_d, N*N*sizeof(Vector3)));

        //Sphere* scene_d;
        //CHECK_CUDA_ERRORS(cudaMalloc(&scene_d, num_spheres*sizeof(Sphere)));
        //CHECK_CUDA_ERRORS(cudaMemcpy(scene_d,scene_h,num_spheres*sizeof(Sphere),cudaMemcpyHostToDevice));

        // allocate random state
        curandState* rand_state_d;
        CHECK_CUDA_ERRORS(cudaMalloc((void **)&rand_state_d, N*N*sizeof(curandState)));

        const dim3 threads(32, 32);
        const dim3 blocks(
            N/threads.x,
            N/threads.y
        );

        Timer timer;
        tick(timer);

        // *NOTE - this takes almost 5 ms, so see if we can reuse this
        // state when attempting RT
        World** world_d;
        CHECK_CUDA_ERRORS(cudaMalloc(&world_d,sizeof(World**)));
        //WorldCreate<<<1,1>>>(world_d);
        RenderInit<<<blocks, threads>>>(N, N, world_d, rand_state_d);

        CHECK_CUDA_ERRORS(cudaGetLastError());
        CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

        Render<<<blocks, threads>>>(img_d, cam, world_d, rand_state_d);
        CHECK_CUDA_ERRORS(cudaGetLastError());
        CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
        deltaT = tick(timer);

        RenderCleanup<<<blocks, threads>>>(world_d);
        world_d = nullptr;

        // Copy data back to host.
        CHECK_CUDA_ERRORS(cudaMemcpy(img_h, img_d, N*N*sizeof(Vector3), cudaMemcpyDeviceToHost));

        // Free memory.
        CHECK_CUDA_ERRORS(cudaFree(img_d));

        // Optional in a single threaded context, but use included for
        // demo purposes.
        cudaDeviceReset();
    }

    std::cout << "Finished. Took: " << 1000.*deltaT << " milliseconds.\n";

    WritePPM("test_out.ppm", img_h, N);
    delete[] img_h;

    //delete that spheres

}