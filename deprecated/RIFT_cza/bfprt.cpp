#include "bfprt.h"
#include <iostream>
#include <vector>


void BFPRT::swap(double &a, double &b)//交换函数
{
	double temp = a;
	a = b;
	b = temp;
}

void BFPRT::insertSort(std::vector<double>&a, int l, int r)//插入排序
{
	if (l = r)
		return;
	for (int i = l + 1; i <= r; ++i)
	{
		for (int j = i; j > l; --j)
		{
			if (a[j - 1] > a[j])
				swap(a[j], a[j - 1]);
		}
	}
}

double BFPRT::getmedium(std::vector<double>&a, int l, int r)//获取数组的上中位数
{
	insertSort(a, l, r);
	return a[l + (r - l) / 2];
}

double BFPRT::getmidofmedium(std::vector<double> &a, int l, int r)//获取中位数组的中位数
{
	int num = r - l + 1;//总数目
	int offset = num % 5 ==0? 0 : 1;
	std::vector<double>medium(num / 5 + offset);//建立中位数数组
	for (int i = 0; i != medium.size(); ++i)
	{
		int ibegin = l + i * 5;//从l开始，5个一组。分组
		int iend = ibegin + 4;
		medium[i] = getmedium(a, ibegin, std::min(iend, r));//把每个小组的中位数填入中位数组中
	}
	return findKthMin(medium, 0, medium.size() - 1,medium.size()/2);//返回中位数组的中位数
}
 
double BFPRT::findKthMin(std::vector<double> &a, int l, int r, int k)//求第k小的数
{
	if (l == r)
		return a[l];//数组中就一个数
	double pvalue = getmidofmedium(a, l, r);//pvalue是数组的中位数组的中位数，
	std::vector<int>pvaluerange = partition(a, l, r,pvalue);//利用pvalue来划分partition
	if (k >= pvaluerange[0] && k <= pvaluerange[1])//pvaluerange数组中的元素是等于pvalue的两个下标。
		return pvalue;
	else if (k < pvaluerange[0])
		return findKthMin(a, l, pvaluerange[0] - 1, k);
	else
		return findKthMin(a, pvaluerange[1] + 1, r,k);
}

std::vector<int> BFPRT::partition(std::vector<double> &a, int l, int r, double pvalue)
{
	int less = l - 1;//小于区的边界
	int more = r+1;//大于区的边界
	int curr = l;
	while (curr< more)
	{
		if (a[curr] < pvalue)//当前值<pvalue
			swap(a[++less], a[curr++]);//当前值与小于区边界值交换，边界往后递增，
		else if (a[curr] > pvalue)//当前值>pvalue
			swap(a[--more], a[curr]);//当前值与大于区边界交换，但是index不用后移，因为换来的不知道是不是大于等于还是小于pvalue
		else
			++curr;//当前值=pvalue，看下一个
	}
	std::vector<int> pvaluerange{ less + 1,more-1 };//返回了等于区的边界下标,此数组中就2个元素
	return pvaluerange;
}

double BFPRT::KthMin()
{
	if (k<1||k>a.size())
		std::cout<<"error"<<std::endl;
	else
		return findKthMin(a,0,a.size()-1,k-1);//第k小的数，在下标上是k-1，
}