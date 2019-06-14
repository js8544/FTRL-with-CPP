#FTRL算法学习总结（2）-jinshang(尚晋)
在上一篇中我们介绍了FTRL算法的理论部分，本篇主要记录用C++实现LR-FTRL模型的过程。

##三、工程实现的技巧

###1. Eigen库

由于C++没有原生的对于线性代数的支持（如矩阵，向量）等，因此我们使用[Eigen](http://eigen.tuxfamily.org/)库来实现线性代数计算。同时，Eigen也提供对于Sparse Vector的支持，为后续的优化提供了基础。

###2. LR类

为了便于解耦，我们将LR模型和FTRL算法分为两个类。其中LR类提供决策函数和梯度计算，而FTRL提供优化和训练函数。

###3. 

##参考资料
1.  [Ad Click Prediction: a View from the Trenches](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf)(Google关于FTRL技术实现的论文)

2.  [FTRL实现](https://github.com/lavizhao/guldan)

3.  [FTRL代码实现](https://www.cnblogs.com/zhangchaoyang/articles/6854175.html)
4.  