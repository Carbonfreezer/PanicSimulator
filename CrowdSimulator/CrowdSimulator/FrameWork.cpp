#include "FrameWork.h"
#include "LogicClass.h"
#include <cstdio>
#include <cassert>
#include "GlobalConstants.h"



void FrameWork::InitializeFramework(LogicClass* usedLogic, const char* windowTitle)
{
	m_usedLogic = usedLogic;

	InitializeGLEWandGLFW(windowTitle);
	InitializeCUDAndOpenGL();

}

void FrameWork::RunCoreLoop()
{
	// Game loop
	double lastTime = glfwGetTime();
	double timeDifference = 0.0;
	while (!glfwWindowShouldClose(m_window))
	{
		// Check if any events have been activiated (key pressed, mouse moved etc.) and call corresponding response functions
		glfwPollEvents();
		

		uchar4* deviceMem;
		size_t num_bytes;
		cudaGraphicsMapResources(1, &m_cuda_pixel_buffer_object);
		cudaGraphicsResourceGetMappedPointer((void **)&deviceMem, &num_bytes, m_cuda_pixel_buffer_object);

		m_usedLogic->UpdateSystem(deviceMem, timeDifference);

		cudaGraphicsUnmapResources(1, &m_cuda_pixel_buffer_object);

		// Render
		// Clear the colorbuffer
		glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		glRasterPos2i(-1, -1);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pixelBuffer);
		glDrawPixels(gScreenResolution, gScreenResolution, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		// Swap the screen buffers
		glfwSwapBuffers(m_window);

		double currentTime = glfwGetTime();
		timeDifference = currentTime - lastTime;
		lastTime = currentTime;
	}
}

void FrameWork::ShutdownFramework()
{
	cudaDeviceReset();
	glfwTerminate();
	
}

void FrameWork::InitializeGLEWandGLFW(const char* windowTitle)
{
	// Init GLFW
	glfwInit();

	// Set all the required options for GLFW
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	// Create a GLFWwindow object that we can use for GLFW's functions
	m_window = glfwCreateWindow(gScreenResolution, gScreenResolution, windowTitle, nullptr, nullptr);

	glfwMakeContextCurrent(m_window);

	
	
	int screenWidth, screenHeight;
	glfwGetFramebufferSize(m_window, &screenWidth, &screenHeight);

	assert(screenHeight == gScreenResolution);
	assert(screenWidth == gScreenResolution);

	// Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions
	glewExperimental = GL_TRUE;
	// Initialize GLEW to setup the OpenGL Function pointers

	if (GLEW_OK != glewInit())
		printf("Failed to initialize GLEW");

}

void FrameWork::InitializeCUDAndOpenGL()
{
	glViewport(0, 0, gScreenResolution, gScreenResolution);
	glGenBuffers(1, &m_pixelBuffer);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pixelBuffer);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, gScreenResolution * gScreenResolution * 4, 0, GL_STREAM_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	// register with cuda.
	const cudaError_t result = cudaGraphicsGLRegisterBuffer(&m_cuda_pixel_buffer_object, m_pixelBuffer, cudaGraphicsMapFlagsWriteDiscard);
	assert(result == cudaSuccess);
}
