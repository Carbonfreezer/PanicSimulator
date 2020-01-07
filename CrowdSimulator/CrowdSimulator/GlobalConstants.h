#pragma once



// Defines required because CUDA can not deal with const floats.

/**
 * \brief The number of blocks we use in every dimension  
 */
#define gNumOfBlocks 9

/**
 * \brief The extension of a block (should be x,y dimension of the block)
 */
#define gBlockSize 32

/**
 * \brief The grid the simulation is actually running on. 
 */
#define gGridSizeInternal (gNumOfBlocks * gBlockSize)

/**
 * \brief The grid size with one extra pixel at the border to avoid special treatments of overlapping. 
 */
#define gGridSizeExternal  (gGridSizeInternal + 2)

/**
 * \brief The amount of pixels we use in visualization for the cell of a grid.
 */
#define gPixelsPerCell  5

/**
 * \brief The resulting screen resolution we get.
 */
#define gScreenResolution  (gGridSizeInternal * gPixelsPerCell)

/**
 * \brief The distance a cell size has in real world (meters).
 */
#define gCellSize  0.5f



/**
 * \brief The maximum walking velocity we have (meters / second). 
 */
#define gMaximumWalkingVelocity  1.34f


/**
 * \brief The maximum density in peoples / sqm. 
 */
#define gMaximumDensity 5.5f


/**
 * \brief The maximum step size we tolerate for the continuity euqatrion solver. 
 */
#define gMaximumStepsizeContinuitySolver 0.010f


/**
 * \brief The maximum error we tolerate on a per pixel basis for the eikonal equation.
 */
#define  gMaximalGodunovError 1.0f


/**
 * \brief The maximum  iterations we intend to cover for the eikonal equation. 
 */
#define gMaximumIterationsGodunov 200


/**
 * \brief The amount of low pass filter iterations we want to do on the velocity.
 */
#define gLowPassFilterVelocity 4


/**
 * \brief The amount of low pass filter operations we want to do on the time.
 */
#define gLowPassFilterTime 1