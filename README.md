# computer_version
本项目主要用于NWPU 机器视觉与人工智能作业，在类`image_processor`中，依靠OpenCV库进行程序编写。

## 依赖：

OpenCV

## 实现功能

对图片进行Resize（发懒没写放大好像）

图片添加高斯噪声，椒盐噪声

图片进行高斯滤波，中值滤波

计算图片滤波前后的SNR

对图片依靠`Sobel`算子进行边缘检测

对图片依靠`Harris`进行角点检测

对图片进行NMS

对图片进行标定
