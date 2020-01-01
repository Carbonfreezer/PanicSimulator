#pragma once

// Float array as used on the graphics card.
struct FloatArray
{
	FloatArray() : m_array(NULL), m_stride(0){}
	float* m_array;
	size_t m_stride;
};

struct UnsignedArray
{
	UnsignedArray() : m_array(NULL), m_stride(0){}
	unsigned int* m_array;
	size_t m_stride;
};
