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
<thead><tr><th> </th><th>图像</th><th>说明</th></tr></thead>
        <tr>
            <td>输入</td>
            <td><a href=""><img width="100%" style="max-width: 30%;max-height:30%;" alt="" src="https://github.com/Liang-yc/lgp_mura_detector/blob/master/gt1.JPG" ></a></td>
            <td></a></td>
        </tr>
 
 
 <tr>
<td>裁剪</td>
<td><img width="100%" style="max-width: 30%;max-height:30%;" alt="" src="https://github.com/Liang-yc/lgp_mura_detector/blob/master/1.BMP"></td>
<td></td>
</tr>

 <tr>
<td>直方图均衡化</td>
<td><img width="100%" style="max-width: 30%;max-height:30%;" alt="" src="https://github.com/Liang-yc/images4readme/blob/master/lgp_mura_detector/%E5%9D%87%E8%A1%A1%E5%8C%96.jpg"></td>
<td></td>
</tr>

<tr>
<td>二值化</td>
<td><img width="100%" style="max-width: 30%;max-height:30%;" alt="" src="https://github.com/Liang-yc/images4readme/blob/master/lgp_mura_detector/%E4%BA%8C%E5%80%BC%E5%8C%96.jpg"></td>
<td></td>
</tr>

<tr>
<td>膨胀腐蚀</td>
<td><img width="100%" style="max-width: 30%;max-height:30%;" alt="" src="https://github.com/Liang-yc/images4readme/tree/master/lgp_mura_detector/膨胀腐蚀.jpg"></td>
<td></td>
</tr>

<tr>
<td>连通区域检测</td>
<td><img width="100%" style="max-width: 30%;max-height:30%;" alt="" src="https://github.com/Liang-yc/images4readme/blob/master/lgp_mura_detector/%E8%BF%9E%E9%80%9A%E5%9B%BE.jpg"></td>
<td></td>
</tr>

<tr>
<td>输出</td>
<td><img width="100%" style="max-width: 30%;max-height:30%;" alt="" src="https://github.com/Liang-yc/images4readme/blob/master/lgp_mura_detector/%E6%A3%80%E6%B5%8B%E7%BB%93%E6%9E%9C.jpg"></td>
<td></td>
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
