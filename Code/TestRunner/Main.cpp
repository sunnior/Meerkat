#include "UnitTestPP.h"
#include "Common/Meerkat.h"

int main()
{
	DeepLearning::Initialize();
	UnitTest::RunAllTests();
	DeepLearning::Finalize();
}