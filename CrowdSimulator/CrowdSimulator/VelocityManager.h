#pragma once
#include <surface_types.h>
#include "TgaReader.h"
#include "TransferHelper.h"
#include "MemoryStructs.h"

// This class is responsible to map the density to the velocity and provide a velocity scalaer field.
class VelocityManager
{
public:

	
	// Generate velocity field.
	void GenerateVelocityField();
	// Sets the file with the walls, only we know the wall information.
	void SetWallFile(const char* wallFilename);
	// Updates the velocity field with the density field handed over.
	void UpdateVelocityField(FloatArray density);

	// Obtains the velocity information.
	FloatArray GetVelocityField() { return m_velocityField; }
	
	// Gets the walls, are also used in the density manager to prevent people getting stuck in walls.
	UnsignedArray GetWallInformation() { return  m_wallInformation; }

	// As we are the only instance that has the wall information we can apply it here.
	void ApplyWallVisualization(uchar4* textureMemory, uchar4 colorToApply);
	// Generates a visualization of the velocity field.
	void GenerateVelocityVisualization(uchar4* textureMemory, float isoLineDistance);
	
private:
	FloatArray m_velocityField;
	UnsignedArray m_wallInformation;

	TgaReader m_wallReader;
	TransferHelper m_helperTransfer;
};
