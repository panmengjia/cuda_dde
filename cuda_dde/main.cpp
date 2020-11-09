#include "main.h"


int main(int argc,char** argv)
{
	double time = (double)getTickCount();
	mainfft();
	time = (getTickCount() - time)*1000 / getTickFrequency();
	cout << "time = " << time <<"ms"<< endl;

	return 0;
}