# 发布到GitHub

以 `face_compare` 的内容作为仓库根目录，应直接包含README、main.py、requirements和 `.github/`。CI放在二级目录不会被GitHub正常识别。

```sh
python tools/build_source_archive.py
```

生成 `dist/FaceCompare-0.4.0-source.zip` 和SHA-256。文件白名单排除本地照片、数据库、模型、环境、真实实验清单和原始测量结果。推荐解压到新目录后发布，避免原工作目录的个人数据和历史进入仓库。

在GitHub创建空仓库，建议名 `FaceCompare`，description为：

> Local face comparison desktop app with YuNet, SFace, unknown rejection and reproducible benchmarks.

在解压的源码根目录执行，将URL替换为自己的仓库地址：

```sh
git init -b main
git add .
git diff --cached --stat
git ls-files
git commit -m "Prepare FaceCompare desktop application"
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

提交前核对暂存清单没有gallery.db、照片或onnx。`.gitignore`不能移除已跟踪文件和历史；使用源码包创建新仓库可避免沿用原历史。本次本地修改尚未创建远程仓库或push。

首页已有中英文说明、真实空库UI预览、架构、使用和实验说明。技术栈徽章不代表CI已通过。上传后查看Actions矩阵，获得实际仓库URL后再添加对应CI状态徽章。

公开演示使用参与者同意发布的素材并移除姓名、编号和路径。从源码包的新目录按README安装、下载模型、检查和启动；首次显示空库，由使用者本地录入。当前zip为源码包，EXE打包需另行验证Qt与目标设备。
