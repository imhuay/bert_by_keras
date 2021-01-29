包管理工具常用命令
===


Index
---
<!-- TOC -->

- [pip](#pip)
    - [帮助](#帮助)
    - [换源](#换源)
        - [linux](#linux)

<!-- /TOC -->


## pip

### 帮助
```shell
# 帮助
pip -h

# 具体命令帮助
pip install -h
pip config -h
```

### 换源
#### linux
```shell
# 创建配置文件
## linux
mkdir ~/.pip
vim ~/.pip/pip.conf
## windows
%HOMEPATH%/pip/pip.ini

# 设置源
[global]
index-url = http://pypi.douban.com/simple
trusted-host = pypi.douban.com

# 常用源
## 阿里源
http://mirrors.aliyun.com/pypi/simple/
## 豆瓣源
http://pypi.douban.com/simple
## 清华源
https://pypi.tuna.tsinghua.edu.cn/simple
```
