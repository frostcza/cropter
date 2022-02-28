#include "bfprt.h"
#include <iostream>
#include <vector>
 
int main()
{
	std::vector<double>a{ 0.0,1.0,2.0,4.0,53.0,8.0,7.0,10.0,33.0,22.0,100.0 };
	BFPRT b(a,4);
	std::cout<<b.KthMin()<<std::endl;
	return 0;
}
