[comment]: <> (# DPV-SLAM)

<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center"> DPV-SLAM (复现及中文注释版~仅供个人学习记录用)
  </h1>

[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center">
  <a href="https://arxiv.org/pdf/2408.01654" target="_blank">Paper</a> 
  | <a href="https://github.com/princeton-vl/DPVO" target="_blank">Original Github Page</a>
  | <a href="https://github.com/KwanWaiPang/DPVO_comment" target="_blank">DPVO的配置与测试</a>
  </h3>
  <div align="center"></div>

# 配置过程记录
* 下载源码并配置conda环境。虽然之前已经配置过dpvo的环境了，但是看yaml代码似乎变换比较大，重新配置为dpv_slam
~~~
git clone https://github.com/princeton-vl/DPVO.git --recursive

conda env create -f environment.yml
conda activate dpv_slam
~~~

* 下载安装eigen库以及安装DPVO包
~~~
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

# install DPVO
pip install .

#模型直接用回之前的试试
地址为：
/home/gwp/DPVO/dpvo.pth
数据地址为：
/home/gwp/DPVO/movies/
~~~