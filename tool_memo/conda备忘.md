包管理工具常用命令
===


Index
---
<!-- TOC -->

- [帮助](#帮助)
- [基础命令](#基础命令)
- [环境管理](#环境管理)
- [换源](#换源)

<!-- /TOC -->


### 帮助
```shell
# 帮助概览
conda -h

# 具体命令帮助
conda install -h
```

### 基础命令
```shell
# 切换环境
## window
activate env_name
## linux
source activate env_name

# 浏览包
conda list

# 安装包
conda 
```

### 环境管理
```shell
# 创建环境
conda create -n py36 anaconda python=3.6
conda create -n huay_ore python=3

# 指定目录创建环境
conda create -p ~/spath/py36 anaconda python=3.6

# 克隆/备份环境
conda create --name dst --clone src

# 删除环境
conda remove --name myenv --all
```

### 换源
```shell
# 新增源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes

# 移除源
conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

# 常用源
## 清华源
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
## USTC 源
https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
```
