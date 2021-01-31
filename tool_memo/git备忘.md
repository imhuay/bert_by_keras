git 常用命令
===

Index
---
<!-- TOC -->

- [参考](#参考)
- [基本命令](#基本命令)
- [删除分支](#删除分支)
- [推送分支](#推送分支)
- [撤回上次 commit](#撤回上次-commit)
- [删除已提交文件/文件夹](#删除已提交文件文件夹)
- [恢复已删除的文件](#恢复已删除的文件)
- [ssh key 相关](#ssh-key-相关)
- [`git subtree` 基本使用](#git-subtree-基本使用)
    - [重新关联子仓库](#重新关联子仓库)
- [修改 commit 的 author 信息](#修改-commit-的-author-信息)

<!-- /TOC -->

### 参考
> https://git-scm.com/book/zh

### 基本命令
```shell
# 初始化，新建本地仓库时使用
git init

# 暂存
git add <path>  # 暂存具体文件/文件夹
git add .   # 暂存新文件和被修改的文件，不包括删除的文件
git add -u  # --update，暂存已追踪的文件，即被修改的文件和被删除的文件
git add -A  # --all，全部暂存

# 提交
git commit -m <'提交信息'>
```

### 删除分支
```
# 删除远程分支
git push origin --delete 分支名
```

### 推送分支
```
# 推送本地分支到远程分支
git push origin 本地分支名:远程分支名
```

### 撤回上次 commit
```
git reset --soft HEAD~1 
-- 撤回最近一次的commit（撤销commit，不撤销git add）

git reset --mixed HEAD~1 
-- 撤回最近一次的commit（撤销commit，撤销git add）

git reset --hard HEAD~1 
-- 撤回最近一次的commit（撤销commit，撤销git add,还原改动的代码）
```

### 删除已提交文件/文件夹
```
# 删除暂存区或分支上的文件，但是工作区还需要这个文件，后续会添加到 .gitignore
# 文件变为未跟踪的状态
git rm --cache <filepath>
git rm -r --cache <dirpath>


# 删除暂存区或分支上的文件，工作区也不需要这个文件
git rm <filepath>
git rm -r <dirpath>


# 不显示移除的文件，当文件夹中文件太多时使用
git rm -r -q --cache <dirpath>
```

### 恢复已删除的文件

**方法 1**：记得文件名
```shell
# 查看删除文件的 commit_id
git log -- [file]

# 恢复文件
git checkout commit_id [file]
```


### ssh key 相关
- ssh key 是远程仓库识别用户身份的依据；

- 如果是通过 ssh 与远程仓库交互，且是第一次在本机执行，则需要先生成 ssh key，然后将**公钥**添加到远程仓库中，以识别用户身份；
    ```shell
    # 生成 ssh key
    ssh-keygen -t rsa -C "邮箱地址"  # 也可以直接执行 ssh-keygen
    ```

- ssh-keygen 会先确认密钥的存储位置（默认是 ~/.ssh/id_rsa）；
- 然后它会要求你输入两次密钥口令。如果你不想在使用密钥时输入口令，将其留空即可（输入 Enter 跳过）。
    ```
    Generating public/private rsa key pair.
    Enter file in which to save the key (/home/sankuai/.ssh/id_rsa):
    Enter passphrase (empty for no passphrase):
    Enter same passphrase again:
    Your identification has been saved in /home/sankuai/.ssh/id_rsa.
    Your public key has been saved in /home/sankuai/.ssh/id_rsa.pub.
    The key fingerprint is:
    SHA256:MUFCejj+NzM47HIQWPuC1l8QrBT0eagQxNpqZu4+hMY huayang04@meituan.com
    The key's randomart image is:
    +---[RSA 2048]----+
    | oo.o+o.o        |
    |  ..o+o+ .       |
    | o.++o=.+        |
    |. oo++.. o       |
    |o. ooo .S        |
    |oEo +o...        |
    |B.   +=.=        |
    | o  ..oo +       |
    |oo.  o.          |
    +----[SHA256]-----+
    ```

- 最后通过 `cat ~/.ssh/id_rsa.pub` 查看生成的公钥，并添加到远程仓库。

### `git subtree` 基本使用
> git subtree教程 - SegmentFault | https://segmentfault.com/a/1190000012002151

- 示例仓库
    ```
    # 父仓库：https://github.com/test/photoshop.git
    photoshop
        |
        |-- photoshop.c
        |-- photoshop.h
        |-- main.c
        \-- README.md

    # 子仓库：https://github.com/test/libpng.git
    libpng
        |
        |-- libpng.c
        |-- libpng.h
        \-- README.md
    ```

1. 建立父子仓库之间的联系

    ```
    > git remote add -f libpng https://github.com/test/libpng.git
    -- 这也相当于给子仓库的路径加一个别名

    # 添加后，可以如下简写
    > git subtree add --prefix=sub/libpng libpng master --squash
    > git subtree pull --prefix=sub/libpng libpng master --squash
    > git subtree push --prefix=sub/libpng libpng master

    # 添加完成后，父版本库中就有两个远程地址了
    > git remote show
    ```

2. 在父仓库中新增子仓库 `git subtree add`
    
    ```
    > git subtree add --prefix=sub/libpng libpng master --squash
    # squash参数表示不拉取历史信息，而只生成一条commit信息。

    # 更新后父仓库结构
    photoshop
        |
        |-- sub/
        |   |
        |   \-- libpng/
        |       |
        |       |-- libpng.c
        |       |-- libpng.h
        |       \-- README.md
        |
        |-- photoshop.c
        |-- photoshop.h
        |-- main.c
        \-- README.md
    ```

    现在 libpng 对其他项目人员来说是透明的：
    当你 `git clone` 或者 `git pull` 的时候，你拉取到的是整个 photoshop（包括libpng，libpng 就相当于 photoshop 里的一个普通目录）；
    当你在父仓库中修改了 libpng 里的内容后执行 `git push`，你将会把修改 `push` 到 photoshop 上，而不是 libpng。

3. 从子仓库拉取更新 `git subtree pull`

    如果子仓库更新了，需要从父仓库拉取这个更新：

    ```
    > git subtree pull --prefix=sub/libpng libpng master --squash
    ```

4. 推送父仓库的更新到子仓库 `git subtree push`

    ```
    > git subtree push --prefix=sub/libpng libpng master
    ```
    
#### 重新关联子仓库
- 比如子仓库A改名并修改了远程地址，且子仓库A还有自己的子仓库B；
- 重新关联前，确保已经 push 最新结果；
- 目录结构
    ```
    father
        |
        |-- A/
        |   |
        |   |-- README.md
        |   \-- B/
        |       |
        |       \-- README.md
        |
        |-- main.c
        \-- README.md
    ```

1. 重命名远程仓库，并修改地址

    ```
    > git remote rename <old_name_a> <new_name_a>
    > git remote set-url <new_name_a> <new_url_a>
    ```

2. 删除当前子仓库A文件夹（其中包含子仓库B），并 commit

3. 重新 add 子仓库A（自动 commit）

    ```
    > git subtree add --prefix=<new_prefix> <new_name_a> master --squash
    ```

4. 删除子仓库A下的子仓库B，并 commit，否则父仓库不能关联子仓库B（提示prefix已存在）

5. 重新 add 子仓库B（因为前缀变了，所以 b 也要重新 add）
    ```
    > git subtree add --prefix=<new_prefix>/b <name_b> master --squash
    ```

6. 完成

### 修改 commit 的 author 信息
> 如何修改git commit的author信息 - 咸咸海风 - 博客园 | https://www.cnblogs.com/651434092qq/p/11015901.html