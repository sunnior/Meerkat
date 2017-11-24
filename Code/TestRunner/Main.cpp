#include "UnitTestPP.h"
#include "Common/Meerkat.h"

int main()
{
	DeepLearning::Init();
	return UnitTest::RunAllTests();
}