#include "FrameWork.h"
#include "CheckerTest.h"
#include "FlagTest.h"
#include "IsoLineTest.h"
#include "EikonalTest.h"
#include "VelocityTest.h"
#include "GradientTest.h"
#include "ContinuityEquationTest.h"
#include "SimulationCore.h"
#include <cassert>
#include <cstring>

static FrameWork gBaseFrameWork;
static SimulationCore gSimulationCore;


static CheckerTest gChecker;
static FlagTest gFlagTest;
static IsoLineTest gIsoLineTest;
static EikonalTest gIconal;
static VelocityTest gVelocity;
static GradientTest gGradientTest;
static ContinuityEquationTest gContinuityTest;

/**
 * \brief Function to use if we want to be able to run the test suite for the different components.
 * \param choice Selector of test to run
 * \param title The title window we want to generate.
 * \param usedLogic The pointer with the used logic we return.
 * \param fileNames the structure with the filenames we want to load.
 */
void PrepareTestSuite(char choice, const char*& title, LogicClass*& usedLogic, BaseFileNames& fileNames)
{
	
	
	
	switch(choice)
	{
	case '0':
		title = "Checker Test";
		usedLogic = &gChecker;
		break;
	case '1':
		title = "Visualization Test";
		usedLogic = &gFlagTest;
		fileNames.m_wallFile = "Picture.tga";
		fileNames.m_initialDensityFile = "Gradient.tga";
		break;
	case '2':
		title = "Iso Line Test 1";
		usedLogic = &gIsoLineTest;
		fileNames.m_initialDensityFile = "Gradient.tga";
		break;
	case '3':
		title = "Iso Line Test 2";
		usedLogic = &gIsoLineTest;
		fileNames.m_initialDensityFile = "BigCircle.tga";
		break;
	case '4':
		title = "Eikonal Test 1";
		usedLogic = &gIconal;
		fileNames.m_targetFile = "SourcePoint.tga";
		break;
	case '5':
		title = "Eikonal Test 2";
		usedLogic = &gIconal;
		fileNames.m_targetFile = "SourcePoint.tga";
		fileNames.m_wallFile = "WallComplex.tga";
		break;
	case '6':
		title = "Eikonal Test 3";
		usedLogic = &gIconal;
		fileNames.m_targetFile = "SourceLine.tga";
		break;
	case '7':
		title = "Eikonal Test 4";
		usedLogic = &gIconal;
		fileNames.m_targetFile = "SourceLine.tga";
		fileNames.m_wallFile = "WallSlit.tga";
		break;
	case '8':
		title = "Eikonal Test 5";
		usedLogic = &gIconal;
		fileNames.m_targetFile = "SourceLine.tga";
		fileNames.m_wallFile = "WallDoubleSlit.tga";
		break;
	case '9':
		title = "Eikonal Test 6";
		usedLogic = &gIconal;
		fileNames.m_targetFile = "SourceLine.tga";
		fileNames.m_wallFile = "WallComplex.tga";
		break;
	case 'A':
		title = "Eikonal Test 7";
		usedLogic = &gIconal;
		fileNames.m_targetFile = "SourceDoublePoint.tga";
		break;
	case 'B':
		title = "Eikonal Test 8";
		usedLogic = &gIconal;
		fileNames.m_targetFile = "SourceDoublePoint.tga";
		fileNames.m_wallFile = "WallSlit.tga";
		break;
	case 'C':
		title = "Eikonal Test 9";
		usedLogic = &gIconal;
		fileNames.m_targetFile = "SourceDoublePoint.tga";
		fileNames.m_wallFile = "WallDoubleSlit.tga";
		break;
	case 'D':
		title = "Eikonal Test 10";
		usedLogic = &gIconal;
		fileNames.m_targetFile = "SourceDoublePoint.tga";
		fileNames.m_wallFile = "WallComplex.tga";
		break;
	case 'E':
		title = "Eikonal Test 11";
		usedLogic = &gIconal;
		gIconal.SetMaximumTime(300.0f);
		fileNames.m_targetFile = "SourceLine.tga";
		fileNames.m_wallFile = "WallConvoluted.tga";
		break;
	case 'F':
		title = "Velocity Test 1";
		usedLogic = &gVelocity;
		fileNames.m_wallFile = "WallComplex.tga";
		fileNames.m_initialDensityFile = "HorrGradient.tga";
		break;
	case 'G':
		title = "Eikonal Density 1";
		usedLogic = &gIconal;
		gIconal.SetIsoLineDistance(5.0f);
		fileNames.m_targetFile = "SourceLine.tga";
		fileNames.m_wallFile = "WallSlit.tga";
		fileNames.m_initialDensityFile = "SpawnAreaSimple.tga";
		break;

	case 'H':
		title = "Eikonal Density 2";
		usedLogic = &gIconal;
		gIconal.SetIsoLineDistance(5.0f);
		fileNames.m_targetFile = "SourceLine.tga";
		fileNames.m_wallFile = "WallSlit.tga";
		fileNames.m_initialDensityFile = "SpawnAreaSimple2.tga";
		break;

	case 'I':
		title = "Eikonal Density 3";
		usedLogic = &gIconal;
		gIconal.SetIsoLineDistance(7.0f);
		fileNames.m_targetFile = "SourceLine.tga";
		fileNames.m_wallFile = "WallSlit.tga";
		fileNames.m_initialDensityFile = "SpawnAreaCircular.tga";
		break;

	case 'J':
		title = "Eikonal Density 4";
		usedLogic = &gIconal;
		gIconal.SetMaximumTime(300.0f);
		gIconal.SetIsoLineDistance(7.0f);
		fileNames.m_targetFile = "SourceLine.tga";
		fileNames.m_wallFile = "WallConvoluted.tga";
		fileNames.m_initialDensityFile = "SpawnAreaCircular.tga";
		break;
	case 'K':
		title = "Velocity Test 2";
		gVelocity.SetStridesOnGradient(5);
		usedLogic = &gVelocity;
		fileNames.m_wallFile = "WallConvoluted.tga";
		fileNames.m_initialDensityFile = "SpawnAreaCircular.tga";
		break;
	case 'L':
		title = "Gradient Test 1";
		usedLogic = &gGradientTest;
		gGradientTest.PrepareTest(0);
		fileNames.m_wallFile = "WallConvoluted.tga";
		break;
	case 'M':
		title = "Gradient Test 2";
		usedLogic = &gGradientTest;
		gGradientTest.PrepareTest(1);
		fileNames.m_wallFile = "WallConvoluted.tga";
		break;
	case 'N':
		title = "Continuity Test 1";
		usedLogic = &gContinuityTest;
		gContinuityTest.PrepareTest(0);
		fileNames.m_initialDensityFile = "Star.tga";
		break;
	case 'O':
		title = "Continuity Test 2";
		usedLogic = &gContinuityTest;
		gContinuityTest.PrepareTest(1);
		fileNames.m_initialDensityFile = "Star.tga";
		break;
	case 'P':
		title = "Continuity Test 3";
		usedLogic = &gContinuityTest;
		gContinuityTest.PrepareTest(2);
		fileNames.m_initialDensityFile = "Star.tga";
		break;
	case 'Q':
		title = "Continuity Test 4";
		usedLogic = &gContinuityTest;
		gContinuityTest.PrepareTest(3);
		fileNames.m_initialDensityFile = "Star.tga";
		break;
	case 'R':
		title = "Continuity Test 5";
		usedLogic = &gContinuityTest;
		gContinuityTest.PrepareTest(0);
		fileNames.m_initialDensityFile = "CenterBlob.tga";
		fileNames.m_wallFile = "MicroWall.tga";
		break;
	case 'S':
		title = "Continuity Test 6";
		usedLogic = &gContinuityTest;
		gContinuityTest.PrepareTest(1);
		fileNames.m_initialDensityFile = "CenterBlob.tga";
		fileNames.m_wallFile = "MicroWall.tga";
		break;
	case 'T':
		title = "Continuity Test 7";
		usedLogic = &gContinuityTest;
		gContinuityTest.PrepareTest(2);
		fileNames.m_initialDensityFile = "CenterBlob.tga";
		fileNames.m_wallFile = "MicroWall.tga";
		break;
	case 'U':
		title = "Continuity Test 8";
		usedLogic = &gContinuityTest;
		gContinuityTest.PrepareTest(3);
		fileNames.m_initialDensityFile = "CenterBlob.tga";
		fileNames.m_wallFile = "MicroWall.tga";
		break;
	case 'V':
		title = "Continuity Test 9";
		usedLogic = &gContinuityTest;
		gContinuityTest.PrepareTest(3);
		fileNames.m_initialDensityFile = "TopPixelLine.tga";
		break;
	case 'W':
		title = "Continuity Test 10";
		usedLogic = &gContinuityTest;
		gContinuityTest.PrepareTest(5);
		fileNames.m_initialDensityFile = "CenterBlob.tga";
		break;
	case 'X':
		title = "Continuity Test 11";
		usedLogic = &gContinuityTest;
		gContinuityTest.PrepareTest(4);
		fileNames.m_initialDensityFile = "OuterRing.tga";
		break;

	}
}

void PrepareCoreSimulation(const char* iniFile,  const char*& title, LogicClass*& usedLogic, BaseFileNames& fileNames)
{
	usedLogic = &gSimulationCore;
	fileNames.LoadFilenames(iniFile);
	title = iniFile;
}

int main(int argc, char **argv)
{	
	// The title for the test,
	const char* title = "Missing title";
	LogicClass* usedLogic = NULL;
	BaseFileNames fileNames;

	if (argc != 2)
		return 0;


	if (strlen(argv[1]) == 1)
		PrepareTestSuite(argv[1][0], title, usedLogic, fileNames);
	else
		PrepareCoreSimulation(argv[1], title, usedLogic, fileNames);


	assert(usedLogic);
	if (usedLogic == NULL)
		return 0;
	
	gBaseFrameWork.InitializeFramework(usedLogic, title, fileNames);	
	gBaseFrameWork.RunCoreLoop();
	gBaseFrameWork.ShutdownFramework();
	return 0;
	
}


