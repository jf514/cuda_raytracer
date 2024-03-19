#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H
#pragma once

#include "vector.h"

#include <fstream>
#include <iostream>
#include <random>

Real RealRand(){
    static std::uniform_real_distribution<Real> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

void WritePPM(const std::string& filename, Vector3* img, int N) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing.\n";
        return;
    }

    // PPM header
    outFile << "P3\n" << N << " " << N << "\n255\n";

    // Generate and write pixel data
    for (int j = 0; j < N; ++j) { // Rows
        for (int  i = 0; i < N; ++i) { // Columns
            // Example gradient effect based on position
            int red = static_cast<int>(255 * img[i + N*j].x);
            int green = static_cast<int>(255 * img[i + N*j].y);
            int blue = static_cast<int>(255 * img[i + N*j].z);

            outFile << red << " " << green << " " << blue << "  ";
        }
        outFile << "\n";
    }

    outFile.close();
    std::cout << "PPM file " << filename << " generated successfully.\n";
}

void CheckCuda(){
    cudaDeviceProp prop;
    int device = 0; // Device number, change this if you want to query properties for a different device

    cudaError_t error = cudaGetDeviceProperties(&prop, device);

    if (error != cudaSuccess) {
        std::cerr << "Error getting device properties: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem << " bytes" << std::endl;
    std::cout << "Shared Memory Per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Registers Per Block: " << prop.regsPerBlock << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads Dimension: [" << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;
    std::cout << "Max Grid Size: [" << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
    std::cout << "Clock Rate: " << prop.clockRate << " kHz" << std::endl;
    std::cout << "MultiProcessor Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

}

#endif // COMMON_UTILS_H
