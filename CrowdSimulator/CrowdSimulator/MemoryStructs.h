#pragma once

// Float array as used on the graphics card.
struct FloatArray
{
	FloatArray();
	void FreeArray();

	float* m_array;
	size_t m_stride;
};

struct UnsignedArray
{
	UnsignedArray();
	void FreeArray();
	
	unsigned int* m_array;
	size_t m_stride;
};
