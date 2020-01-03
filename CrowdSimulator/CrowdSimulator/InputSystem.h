#pragma once
#include <GLFW/glfw3.h>



/**
 * \brief Administrates all the different inputs. Al states can be addressed with the appropriate functions.
 * Uses currently the glfw library internally.
 */
class InputSystem
{
public:
	/**
	 * \brief Initializes the input system.
	 * \param window The window we poll for input.
	 */
	void InitializeSystem(GLFWwindow* window);

	/**
	 * \brief Updates the current system.
	 */
	void Update();

	/**
	 * \brief Checks if the user wants to restart the simulation.
	 * \return 
	 */
	bool WasResetToggled() { return m_resetToggled; }

	/**
	 * \brief Asks for the current visualization mode.
	 * \return Visualization mode 0: density, 1: eikonal field and velocity, 2: (later on) crowd pressure.
	 */
	int GetVisualizationMode() { return m_visualizationMode ; }

	/**
	 * \brief Gets the annotation mode.
	 * \return Do we show annotation
	 */
	bool GetAnnotationMode() { return m_annotationMode; }
	
	/**
	 * \brief Explains if the simulation is paused.
	 * \return Returns whether simulation is paused.
	 */
	bool IsSimulationPaused() { return m_isSimulationPaused; }

	/**
	 * \brief Gets the velocity selector from the input ranging from 0 to 3 
	 * \return Velocity indicator.
	 */
	int GetVelocityCounter() { return m_velocityCounter; }

	/**
	 * \brief Indicates if we want to shoe the eikonal solution.
	 * \return If we want to show the eikonal solution.
	 */
	bool ShowsEikonalSolution() { return m_showsEikonalSolution; }

	/**
	 * \brief Returns the distance between isolines in the Eikonal visualization.
	 * \return distance between iso lines.
	 */
	float GetIsoLineDistance() { return (float)m_isoLineDistance; }
	
private:
	GLFWwindow* m_window;

	bool m_resetToggled;
	int m_visualizationMode;
	bool m_annotationMode;
	bool m_showsEikonalSolution;
	int m_velocityCounter;
	int m_isoLineDistance;
	
	bool m_wasSpacePressed;
	bool m_wasAPRessed;
	bool m_wasEPressed;
	bool m_isSimulationPaused;
	bool m_wasUpPressed;
	bool m_wasDownPressed;


	bool m_wasLeftPressed;
	bool m_wasRightPressed;

};
