#pragma once

#include "GlobalConstants.h"

#define CUDA_DECORATOR_LOGIC <<<dim3(gNumOfBlocks, gNumOfBlocks), dim3(gBlockSize, gBlockSize)>>>
#define CUDA_DECORATOR_SCREEN <<<dim3(gNumOfBlocks * gPixelsPerCell, gNumOfBlocks * gPixelsPerCell), dim3(gBlockSize, gBlockSize)>>>
