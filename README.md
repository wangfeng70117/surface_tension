# 在Taichi MPM示例代码的基础上构建SDF、并将SDF转化为Marching Cube构造显示表面



## 背景简介

构建流体表面是流体仿真非常常用的一个技术，构建表面可以分为显示表面和隐式表面，隐式表面通常使用Level Set方法构建，显示表面最简单的方法是Marching Cube方法，显示表面和隐式表面是可以互相转化的。

这个代码里，每个流体粒子创建了一个球形Level Set，然后union每个球形Level Set生成一个大的整个流体区域的Level Set，在生成Level Set的过程中也构建出了SDF符号距离场，使用SDF将隐式Level Set转化为了显示的Marching Cube。

## 成功效果展示

https://forum.taichi.graphics/uploads/default/original/1X/e163379e1f66d011148ea35f7bcc2463b985b7b0.gif

## 整体结构（Optional）



```
-README.MD
-gen_surface.py
```

## 运行方式



Just: `python gen_surface.py`
