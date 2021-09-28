import imageio
import os

gif_images = []

dirs = os.listdir("F:\\代码\python代码\\surface_tension")
for d in dirs:
    if d[0:2] == "00":
        gif_images.append(imageio.imread("F:\\代码\\python代码\\surface_tension" + "\\" + d))  # 读取多张图片
imageio.mimsave("hello.gif", gif_images, fps=90)   # 转化为gif动画
