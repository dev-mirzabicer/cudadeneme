#ifndef CUDA_SETTINGS_H
#define CUDA_SETTINGS_H

// CUDA settings and constants
namespace CudaSettings {
    const int blockSize = 256;
    const int tileSize = 16;  // Tile size for matrix multiplication
}

#endif // CUDA_SETTINGS_H
