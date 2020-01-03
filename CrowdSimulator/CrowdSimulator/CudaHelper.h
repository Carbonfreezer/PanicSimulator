#pragma once

#include "GlobalConstants.h"

/**
 * \brief Grid invocation parameters for allmost all the kernels in the program.
 */
#define CUDA_DECORATOR_LOGIC <<<dim3(gNumOfBlocks, gNumOfBlocks), dim3(gBlockSize, gBlockSize)>>>
