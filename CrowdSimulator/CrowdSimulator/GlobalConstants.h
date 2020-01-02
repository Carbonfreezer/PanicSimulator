#pragma once



// Defines required because CUDA can not deal with const floats.

// The number of blocks we use in every dimension 
#define gNumOfBlocks 9

// The extension of a block
#define gBlockSize 32

// The grid the simulation is actually running on.
#define gGridSizeInternal (gNumOfBlocks * gBlockSize)

// The grid size with one extra pixel at the border to avoid special treatments of overlapping.
#define gGridSizeExternal  (gGridSizeInternal + 2)

// The amount of pixels we use in visualization for the cell of a grid.
#define gPixelsPerCell  5

// The resulting screen resolution we get.
#define gScreenResolution  (gGridSizeInternal * gPixelsPerCell)

// The distance a cell size has in real world (meters).
#define gCellSize  0.5f

// The diagonal cell size.
#define gCellSizeDiagonal (gCellSize * 1.41421f)


// The maximum walking velocity we have (meters / second).
#define gMaximumWalkingVelocity  1.34f


// The maximum density in peoples / sqm.
#define gMaximumDensity 5.5f

// The maximum step size we tolerate for the continuity solver.
#define gMaximumStepsizeContinuitySolver 0.010f

// The maximum error we tolerate on a per pixel basis.
#define  gMaximalGodunovError 1.0f

// The maximum double iterations we intend to cover.
#define gMaximumIterationsGodunov 100