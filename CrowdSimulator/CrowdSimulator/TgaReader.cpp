#include "TgaReader.h"
#include <cstddef>
#include <cstdio>
#include <cassert>


TgaReader::TgaReader()
{
	m_pixels = NULL;
	m_height = 0;
	m_width = 0;
}


TgaReader::~TgaReader()
{
	if (m_pixels != NULL)
		delete [] m_pixels;

	m_pixels = NULL;
}

void TgaReader::ReadFile(const char* fileName)
{
	if (m_pixels != NULL)
		delete m_pixels;
	m_pixels = NULL;
	
	unsigned char header[12] = { 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	unsigned char bpp = 32;
	unsigned char id = 8;
	unsigned short width;
	unsigned short height;
	FILE *fp = NULL;

	fopen_s(&fp, fileName, "rb");

	fread(header, sizeof(unsigned char), 12, fp);
	fread(&width, sizeof(unsigned short), 1, fp);
	fread(&height, sizeof(unsigned short), 1, fp);
	fread(&bpp, sizeof(unsigned char), 1, fp);
	fread(&id, sizeof(unsigned char), 1, fp);

	assert(bpp == 24);

	m_pixels = new unsigned char[width * height * 3];
	fread(m_pixels, 3, width * height, fp);
	fclose(fp);

	m_width = width;
	m_height = height;
}
