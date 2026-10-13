# FaceCompare

**基于 YuNet 与 SFace 的本地人脸识别与分析桌面应用。** 支持USB摄像头、多人逐脸识别、轨迹连续确认、Top-K候选解释，以及阈值标定和可视化实验报告。

[English](README_EN.md) · [操作手册](docs/操作手册.md) · [架构说明](docs/ARCHITECTURE.md) · [实验复现](docs/EXPERIMENTS.md) · [贡献指南](CONTRIBUTING.md)

![Python 3.11–3.12](https://img.shields.io/badge/Python-3.11%E2%80%933.12-3776AB?logo=python&logoColor=white)
![OpenCV 4.12](https://img.shields.io/badge/OpenCV-4.12-5C3EE8?logo=opencv&logoColor=white)
![PySide6](https://img.shields.io/badge/UI-PySide6-41CD52)
![Code license MIT](https://img.shields.io/badge/Code-MIT-blue)

![FaceCompare空库启动界面](docs/assets/ui-preview.png)

*预览由真实应用在临时空库中离屏渲染，无摄像头输入，无真实人脸或个人信息。*

## 功能

- 摄像头实时预览、抓拍与连续识别，也可通过命令行识别单张图片。
- 多人模式一次检测、逐脸质量检查与匹配；轻量IoU轨迹关联，连续多次匹配后确认姓名。
- Top-K身份候选解释，带轨迹编号的分析快照与实时画面分开显示。
- 未知人员拒识；区分无脸、多脸、人脸过小、模糊和亮度异常。
- 人员录入、补充样本、预览、删除与SQLite持久化，模板即时更新。
- 推理和录入使用后台线程；旧设备、旧库版本及过期结果不展示。
- 固定模型SHA-256校验，启动配置验证，支持中文图片路径。
- 验证集阈值标定、图片/视频性能对照、原始CSV与汇总JSON。
- 预测CSV可生成分数分布、开放集工作点曲线和按场景的指标报告。
- 回归测试、Ruff检查、GitHub Actions以及排除本地数据的源码归档。
- v0.4：可重试图片清理、库完整性检查、只读实验与冻结核验、阶段/队列 p50/p95，以及项目检查入口。

本轮实际修复与限制见 [工程交付记录](docs/HARDENING.md)、[性能实测](docs/PERFORMANCE.md) 和 [实验计划](refine-logs/EXPERIMENT_PLAN.md)。

## 快速开始

推荐 Windows 10/11、Python 3.11、普通 CPU 和 USB 摄像头。在本目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --download-models
python main.py --check
```

模型来自 OpenCV Zoo，下载程序先写临时文件，检查文件大小及预先固定的 SHA-256，再原子替换并生成本地模型清单。已有模型同样需要校验。模型二进制不纳入源代码版本控制。

默认使用CPU推理。完成首次下载后，采集、录入和识别均在本机运行。Linux激活命令为 `source .venv/bin/activate`。摄像头现场运行以Windows为主要目标；CI配置覆盖Windows/Ubuntu的Python 3.11/3.12，其他平台的设备兼容性仍需实机验证。

## 启动与使用

```powershell
python main.py
```

1. 在设备列表选择经预览确认的外置摄像头并开启。
2. 使用“人员录入 / 补充样本”为至少 3 名不同人员分别采集 10–20 张有效图像。
3. “抓拍识别”执行单次比对；“连续识别”按配置频率处理，待处理队列永远只保留最新帧。
4. 未达到阈值时界面显示“未知人员”，无脸、多脸和质量不足会显示各自状态。
5. “标准库管理”可预览并删除人员或样本；所有变化立即发布新的只读模板快照。

初始阈值 `0.42` 只是工程起点，不是最终实验结论。必须使用独立验证集标定，再将锁定后的阈值写回 `config.json`，然后评估测试集。

```powershell
# 不启动窗口或摄像头，识别一张图片
python main.py --image "D:\图片\查询.jpg" --json
python main.py --image "D:\图片\合照.jpg" --multi-face --json

# 使用独立配置，例如分开演示库和实验库
python main.py --config config.local.json
python main.py --version
```

`--image`使用所选配置的标准库，不写识别日志；库不存在时会创建空库。`unknown`、`no_face`等正常处理状态的退出码为0；输入、配置或环境错误返回1。

## 算法与配置

```mermaid
flowchart LR
    A[摄像头或图片] --> B[YuNet 人脸框与五点关键点]
    B --> C{单脸且质量合格}
    C -->|否| D[采集状态提示]
    C -->|是| E[SFace 对齐与特征提取]
    E --> F[L2归一化与身份模板匹配]
    F --> G{最高余弦分达到阈值}
    G -->|是| H[人员姓名]
    G -->|否| I[未知人员]
```

人员模板由归一化样本特征求均值后再次归一化生成。最高分达到阈值才接受身份，余弦相似度不代表身份正确概率。模板按识别模型哈希和预处理版本隔离。

所有相对文件路径都以配置文件所在目录为基准。常用配置如下：

| 配置 | 作用 | 默认值 |
| --- | --- | --- |
| `camera.default_index` | 首选摄像头编号 | `0` |
| `camera.width` / `height` | 请求采集尺寸，实际尺寸由设备决定 | `640` / `480` |
| `camera.recognition_interval_ms` | 连续识别提交间隔 | `600` |
| `camera.result_ttl_ms` | 从采集时刻计算的结果有效期 | `2500` |
| `detection.detect_scale` | 检测缩放，关键点映射回原图 | `0.5` |
| `quality` | 人脸尺寸、清晰度、亮度检查 | 见 `config.json` |
| `matching.cosine_threshold` | 身份接受阈值 | `0.42` |

主画面展示实时视频，右侧单独展示识别快照。关闭或切换设备、变更标准库、结果过期后清除旧结果。影响特征生成的修改需要更新预处理版本并重新建立兼容标准库。

## 多人识别与连续确认

勾选“多人识别 / 跟踪”和“连续识别”。每帧只检测一次，再逐脸检查质量和识别；默认最多处理5张，超过上限会明确记录。左侧“识别分析快照”展示轨迹框，右侧逐脸列表可展开Top-K候选。

默认同一轨迹连续3次匹配同一身份才显示姓名。身份变化、未知、质量失败、缺失、设备/库/模式切换会打断或重置证据；重复抓拍同一帧不会增加确认次数。候选相似度只用于解释，不代表接受身份。

跟踪是原创的轻量IoU基线，适合少量人脸缓慢运动，不具备ByteTrack的运动预测与完整关联流程。快速移动或交叉可能换轨迹号；编号不代表真实身份。连续确认用于减少显示抖动，误接收改善仍需实验。

单人模式和录入仍要求单脸。图片 `--multi-face` 返回独立逐脸结果，不对静态图片声称跨帧确认。

| 配置 | 作用 | 默认值 |
| --- | --- | --- |
| `matching.top_k` | 身份级候选数量 | `3` |
| `tracking.max_faces` | 单帧处理人数上限 | `5` |
| `tracking.iou_threshold` | 关联IoU下限 | `0.3` |
| `tracking.max_gap_ms` | 最大缺失时间 | `2000` |
| `tracking.confirmation_frames` | 连续确认次数 | `3` |

开源项目对照和后续路线见 [项目提升与对照](docs/UPGRADES.md)。

## 数据管理

- 仅采集已明确同意参与实验的人员；图像保存在本机 `data/samples/`，不要提交到公开仓库。
- 注册、验证和最终测试按拍摄会话隔离。同一视频的相邻帧不能拆到不同集合。
- 未知人员不能出现在标准库。动态扩容演示中新录入的人，也不能继续计入最终未知测试。
- 数据库样本同时保存模型哈希和预处理版本；不兼容特征不会进入当前模板快照。

## 离线评估

复制 `experiments/manifests/example.csv` 并填写真实图像。字段含义：

| 字段 | 含义 |
| --- | --- |
| `sample_id` | 唯一样本编号 |
| `image_path` | 相对清单或绝对图像路径 |
| `split` | `validation` 或 `test` |
| `is_known` | 是否属于标准库身份 |
| `expected_person_id` | 已知样本对应数据库人员 ID；未知留空 |
| `scene` | 光照、姿态、眼镜等场景标签 |
| `session_id` | 拍摄会话，用于检查数据隔离 |

在验证集上标定目标有效未知 FAR 不高于 5%：

```powershell
python experiments/evaluate.py experiments/manifests/my_manifest.csv `
  --split validation --calibrate-target-far 0.05 `
  --freeze-output experiments/results/run.freeze.json `
  --output experiments/results/validation
```

将输出阈值写回配置并锁定后运行最终测试：

```powershell
python experiments/evaluate.py experiments/manifests/my_manifest.csv `
  --split test --freeze experiments/results/run.freeze.json --output experiments/results/test
```

输出包括逐样本 `predictions.csv` 和包含计数、全尝试指标、有效查询指标及模型/库版本的 `summary.json`。图像读取失败、无脸、多脸和质量拒绝不会被算作“未知正确拒识”。

清单检查拒绝重复编号、图像内容重复、跨split会话、注册数据重叠，以及标准库中不存在的已知身份。阈值标定只允许 `validation`。复制输出的完整精度阈值到配置，不要只复制界面三位小数。test 强制使用验证阶段的冻结记录；修改模型、标准库、数据、代码或运行环境均会使核验失败。实验库只读且必须已存在；详见 [实验复现说明](docs/EXPERIMENTS.md)。

```powershell
python experiments/report.py experiments/results/test/predictions.csv --threshold 0.42 --output experiments/results/report
```

使用已锁定阈值生成 `analysis.png`、`analysis.json`、`operating_curve.csv` 和 `scene_summary.csv`。曲线表示1:N已知正确识别率与有效未知FAR的关系，区别于两图验证ROC。图表供分析，不能据最终测试曲线反向调阈值；缺少有效未知样本时不生成虚构FAR。

## 时间优化实验

固定图像上的 B0/B1/B2/B4 对照：

```powershell
python experiments/benchmark.py experiments/manifests/my_manifest.csv `
  --split test --repeats 3 --output experiments/results/benchmark
python experiments/plot_results.py `
  experiments/results/benchmark/benchmark_summary.csv `
  --output experiments/results/benchmark/latency.png
```

固定视频上的队列调度对照：

```powershell
python experiments/video_benchmark.py data/evaluation/fixed_video.mp4 `
  --repeats 3 --max-seconds 30 --output experiments/results/video_benchmark
```

- B0：原尺寸检测、逐模板 Python 循环、处理所有排队帧。
- B1：缩小图像检测，框与五点关键点全部映射回原图后再对齐。
- B2：模板矩阵一次性计算余弦相似度。
- B3：队列只保留最新待处理帧，报告丢帧率与端到端时延。
- B4：缩小检测、矩阵匹配、最新帧调度组合。

固定图像脚本报告算法时延；固定视频脚本报告应用入口到结果完成的端到端时延。首次模型加载不混入重复测量，预览 FPS 也不作为识别 FPS。

## 开发与测试

```powershell
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest -m "not integration" -q
python -m pytest -m integration -q
python tools/check_project.py --models
```

测试使用临时数据库和合成数组，覆盖配置、模板匹配、模型下载失败、保存回滚、实验隔离、UI状态及源码包内容。真实模型测试需要已下载的权重，使用空画面，不读取真实人脸或启动摄像头。GitHub Actions在push和PR时运行常规检查；USB拔插等设备测试按操作手册在目标电脑执行。

已完成 [AT&T/ORL公开照片基准](docs/PUBLIC_BENCHMARK.md)：已知60/60正确、未知94/100正确拒识、6次未检出、有效未知误接受0/94；B0→B4算法p50为35.51→29.08 ms。这是独立质量配置下的图片划分，缺少会话标签，不代表默认摄像头泛化性能。工具可重新下载校验数据、建独立库、验证标定并冻结测试。当前未发布跨人群最终识别结果，未实现专门的活体或照片攻击检测。

## 工程结构

```text
app/                 配置、相机线程、检测、质量、特征、匹配、持久化、UI
experiments/         阈值评估、固定图像/视频性能对照、绘图
tests/               不依赖真实人脸数据的单元测试
models/              下载的 ONNX 模型与模型清单
data/                SQLite 数据库和本地人脸样本
docs/                操作与验收说明
tools/               源码归档和无个人数据的预览生成
.github/             CI、Issue模板和PR模板
```

算法路线依据 OpenCV 官方 `FaceDetectorYN` 与 `FaceRecognizerSF` 接口。界面中的数值明确标为余弦相似度，不将它解释为身份正确概率。

## 发布到GitHub

以当前 `face_compare` 目录的内容作为仓库根目录，保留 `.github` 等隐藏文件，确保CI在根目录找到依赖文件。步骤见 [发布说明](docs/PUBLISHING.md)。

```powershell
python tools/build_source_archive.py
```

输出 `dist/FaceCompare-0.4.0-source.zip` 和SHA-256文件。使用文件白名单，不包含本地人脸照片、数据库、模型权重、真实实验清单、虚拟环境或原始测量结果。项目专用 [facecompare-check](skills/facecompare-check/SKILL.md) 与 AGENTS.md 一并提供。

## 来源与许可证

应用源码采用 [MIT License](LICENSE)。模型与依赖遵循各自许可证，见 [第三方说明](THIRD_PARTY_NOTICES.md)。

来源：[OpenCV官方教程](https://docs.opencv.org/4.12.0/d0/dd4/tutorial_dnn_face.html)、[YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)、[SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)。

