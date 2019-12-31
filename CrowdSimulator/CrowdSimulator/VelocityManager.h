#pragma once
#include <surface_types.h>
#include "TgaReader.h"
#include "TransferHelper.h"

// This class is responsible to map the density to the velocity and provide a velocity scalaer field.
class VelocityManager
{
public:
	VelocityManager() { m_velocityField = NULL; m_wallArea = NULL; }

	
	// Generate velocity field.
	void GenerateVelocityField();
	// Sets the file with the walls, only we know the wall information.
	void SetWallFile(const char* wallFilename);
	// Updates the velocity field with the density field handed over.
	void UpdateVelocityField(float* densityField, size_t densityStride);

	// Obtains the velocity information.
	float* GetVelocityField(size_t& velocityStride)
	{
		velocityStride = m_velocityStride;
		return m_velocityField;
	}

	// Gets the walls, are also used in the density manager to prevent people getting stuck in walls.
	unsigned int* GetWallInformation(size_t& wallStride)
	{
		wallStride = m_wallStride;
		return m_wallArea;
	}

	// As we are the only instance that has the wall information we can apply it here.
	void ApplyWallVisualization(uchar4* textureMemory, uchar4 colorToApply);
	// Generates a visualization of the velocity field.
	void GenerateVelocityVisualization(uchar4* textureMemory, float isoLineDistance);
	
private:
	float* m_velocityField;
	size_t m_velocityStride;

	unsigned int* m_wallArea;
	size_t m_wallStride;

	TgaReader m_wallReader;
	TransferHelper m_helperTransfer;
};
