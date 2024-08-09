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
#如果直接用本仓库的代码不需要下载
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

# install DPVO (感觉配置文件有点问题，尝试换回DPVO conda环境。conda activate dpvo)
pip install .

#模型直接用回之前的试试
地址为：
/home/gwp/DPVO/dpvo.pth
数据地址为：
/home/gwp/DPVO/movies/
~~~
* 安装可视化工具
~~~
./Pangolin/scripts/install_prerequisites.sh recommended #前面应该已经安装过的了~
mkdir Pangolin/build && cd Pangolin/build
cmake ..
make -j8
sudo make install
cd ../..
pip install ./DPViewer
~~~
* 对于回环部分
~~~
# 首先需要安装opencv
sudo apt-get install -y libopencv-dev

cd DBoW2
mkdir -p build && cd build
cmake .. # tested with cmake 3.22.1 and gcc/cc 11.4.0 on Ubuntu
make # tested with GNU Make 4.3
sudo make install
cd ../..
pip install ./DPRetrieval
~~~

# 运行测试