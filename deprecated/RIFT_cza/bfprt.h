#ifndef BFPRT_H
#define BFPRT_H
//BFPRT算法是在不排序的情况下，找到第k小，或者第k大的数
//这里以第k小的数为例，进行算法实现
//1、首先对数组进行分组，5个为1组，最后不足5个的为1组，一共有num/5或者1+num/5组
//2、对每组进行插入排序，排好序后，取每个组的中位数，组成中位数数组mediums[n]
//3、然后，求出mediums数组中的上中位数pvalue，这里递归调用BFPRT算法
//求mediums数组中的上中位数，就是求该数组中，第mediums.size()/2个数
//4，此时pvalue，就是得到的划分值。然后进行与快排里面的partition函数一样的划分区域
//小于pvalue的在小于区，等于的在等于区，大于的在大于区。如果要求得第k小的数，在等于区
//也就是说k-1作为下标，在等于区的下标范围内，那么直接返回pvalue，
//如果k-1小于等于区的左边界下标，说明在小于区内，继续partition，同理，大于区内也一样。
//这里，因为确定了在小于区，就不用管大于区，所以递归只走一边，时间复杂度比快排小
 
#include <iostream>
#include <vector>

class BFPRT
{
public:
    BFPRT(std::vector<double> vec, int k):a(vec),k(k){}
    double KthMin();//数组中第k小的数

private:
    void swap(double &a, double &b);//交换函数
    void insertSort(std::vector<double>&a, int l, int r);//插入排序
    double getmedium(std::vector<double>&a, int l, int r);//获取数组的上中位数
    double getmidofmedium(std::vector<double> &a, int l, int r);//获取中位数组的中位数
    double findKthMin(std::vector<double> &a, int l, int r, int k);//求第k小的数
    std::vector<int> partition(std::vector<double> &a, int l, int r, double pvalue);
    
    std::vector<double> a;
    int k;
};

#endif