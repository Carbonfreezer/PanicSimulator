#pragma once

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
class LogicClass;

class FrameWork
{
public:


	void InitializeFramework(LogicClass* usedLogic, const char* windowTitle);
	void RunCoreLoop();
	void ShutdownFramework();

private:
	GLuint m_pixelBuffer;
	cudaGraphicsResource* m_cuda_pixel_buffer_object;
	GLFWwindow *m_window;

	LogicClass* m_usedLogic;


	void InitializeGLEWandGLFW(const char* windowTitle);
	void InitializeCUDAndOpenGL();


};
