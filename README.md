# 导光板缺陷检测
# 背景介绍
------
该项目用于导光板(Light Guide Plate)的缺陷检测,检测的缺陷类型为线缺陷(mura?)。通过对输入图像进行处理，标记图像中存在的缺陷位置。（目前尚未完成）
<br><br>
# 方法
------
该方法包含以下几个步骤：
<br><br>
<table>
<thead><tr><th> </th><th>图像</th>说明</tr></thead>
        <tr>
            <td>输入</td>
            <td><a href=""><img width="100%" style="max-width: 30%;max-height:30%;" alt="" src="https://github.com/Liang-yc/images4readme/blob/master/red_plate.JPG" ></a></td>
            <td><a href=""><img width="100%" style="max-width: 30%;max-height:30%;" alt="" src="https://github.com/Liang-yc/images4readme/blob/master/white_plate.JPG" ></a></td>
        </tr>
</table>
<br>


# 文件
---------
1. `gt1.jpg`,`gt2.jpg`以及`gt3.jpg`为标注图像；
2. `1.BMP`,`2.BMP`以及`3.PNG`为对应图像裁剪后的图像；
3. `detect.cpp`即为本项目代码文件。

# 实验环境
-------
1.硬件环境：Windows 7x64，i3处理器；
2.软件环境：VS2013, openCV 3.0
