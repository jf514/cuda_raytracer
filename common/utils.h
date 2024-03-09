#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H
#pragma once

#include "vector.h"

#include <fstream>
#include <iostream>

void WritePPM(const std::string& filename, Vector3* img, int N) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing.\n";
        return;
    }

    // PPM header
    outFile << "P3\n" << N << " " << N << "\n255\n";

    // Generate and write pixel data
    for (int i = 0; i < N; ++i) { // Rows
        for (int j = 0; j < N; ++j) { // Columns
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

#endif // COMMON_UTILS_H
