---
title: Python环境配置相关
tags: Technical Manual
---
如果在上课时遇到安装库缓慢或环境配置，可以参考下面操作提示：

## 一、国内常用景象资源地址：
``` bash
$ https://pypi.tuna.tsinghua.edu.cn/simple  # 清华大学
$ https://mirrors.aliyun.com/pypi/simple    # 阿里云(推荐)
$ https://mirrors.cloud.tencent.com/pypi/simple # 腾讯云
$ https://pypi.mirrors.ustc.edu.cn/simple/  # 中国科技大学
$ http://pypi.hustunique.com/   # 华中科技大学
$ http://pypi.sdutlinux.org/    # 山东理工大学
$ https://pypi.douban.com/simple # 豆瓣
```
### （临时换源方法）执行命令
``` bash
$ pip install 替换为库名称==替换为版本号（不填表示安装最新版本） -i 国内景象资源
```
### （永久换源方法）执行命令，执行后每次pip不需要-i。
``` bash
$ pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
```
### 导入完毕后在代码中验证(示例)
``` bash
$ import requests


$ print(requests.__version__)
```
---
## 二、环境配置问题：
1、对于Windows系统（macOS无视），下载python安装包后，建议自定义安装到C盘根目录。如果是固态硬盘，推荐安装到D盘，如C盘系统崩溃恢复后，可将python安装包覆盖安装到之前的目录避免重新导包。
2、安装包完成后：
->右键点击“此电脑”选择属性
->高级系统设置
->环境变量
->系统变量中双击Path
->新建2条(注意选择python安装所在的盘符)
->在右侧将两条路径上移到顶
``` bash
$ C:\Python\Python36\Scripts
$ C:\Python\Python36\
```
3、验证安装：Win+R键->cmd->确认
``` bash
$ python --version
```
``` bash
$ pip -V
```
显示对应安装的版本后表明已安装成功。

---
## 三、其他操作
通过下面命令查看当前设置的镜像源：
``` bash
$ pip config list
```
删除全局设置的镜像源:
``` bash
$ pip config unset global.index-url
```
删除用户级别设置的镜像源:
``` bash
$ pip config unset global.index-url --user
```
如果还设置了其他镜像源（如 extra-index-url），也需要一并删除：
``` bash
$ pip config unset global.extra-index-url --user
```
更新 pip 本身：
``` bash
$ pip install --upgrade pip
```
删除当前环境的所有 pip 缓存：
``` bash
$ pip cache purge
```
删除特定包的缓存：
``` bash
$ pip cache remove package_name     # package_name替换为库的名称
```
遇到下载不稳定时，可以设置超时时间：
``` bash
$ pip install package_name --timeout 10     # 将超时时间设置为 10 秒
```
将当前环境的包导出：
``` bash
$ pip freeze > requirements.txt
```
从 requirements.txt 安装指定包：
``` bash
$ pip install -r requirements.txt
```
自定义源时的信任设置：
``` bash
$ pip install package_name --trusted-host pypi.tuna.tsinghua.edu.cn # package_name替换为库的名称
```
## 四、附录
安装wheel文件（xxxx.whl）
``` bash
$ pip install 文件路径/xxxx.whl # 注意文件路径格式
```

More info: whl文件下载方法参考这个网址下面图示[资源地址参考](https://blog.csdn.net/weixin_57950978/article/details/142653359)


