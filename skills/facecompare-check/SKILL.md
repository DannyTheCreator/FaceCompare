---
name: facecompare-check
description: 检查 FaceCompare 本地人脸比对项目的代码修改、实验冻结协议、线程生命周期和源码归档；用于该项目交付前复核，不执行远程发布或真实人脸采集。
---

# FaceCompare 修改复核

从包含 main.py 和 config.json 的项目根目录运行。先读 AGENTS.md 与受修改模块对应文档；以代码为准，保护已有修改。没有 Git 时用白名单工具保存独立名称的源码 ZIP 基线。

按实际改动选择检查：

- 相机/线程：旧 stream、mode generation、gallery version、TTL 结果和错误都应被过滤；pending 队列只保留最新帧；关闭窗口必须等待 QThread 完成。摄像头真机测试与 Qt 离屏测试分开报告。
- 数据：SQLite 删除和 pending_file_deletions 同事务；文件清理失败后应能重试，不能留下已删除身份的内存模板。路径须在 samples_dir 内。检查缺图、孤立图片、特征版本和维度；不自动删除未引用文件。
- 实验：验证/测试复用 service，但 repository 只读，不建库或写日志。SHA-256 检查复制文件，session_id 隔离拍摄；哈希不能识别重新压缩副本。test 必须使用 validation 生成的 --freeze，核验阈值、模型、库内容、数据、代码和环境；开始/结束都核验。
- 性能：复用 experiments/benchmark.py 和 video_benchmark.py。预热移出计时，比较相同输入/硬件/配置，报告 p50/p95、输出变化、丢帧率；预览 FPS、有效显示频率、后台识别吞吐和端到端时延分别解释。

公开图片没有会话标签时，只允许明确标记 split_unit=image、显式 --allow-image-split 的独立探索性基准；保留未知会话事实。使用 tools/prepare_orl.py 时检查官方来源、校验和、固定身份划分、独立数据库和质量配置差异；不能用公开照片实验冒充默认摄像头或会话隔离成绩。方法见 docs/PUBLIC_BENCHMARK.md。

运行 `python tools/check_project.py`，有已下载模型时加 `--models`。该入口禁用第三方 pytest 插件自动加载，不打开相机，使用临时数据；模型缺失时集成测试可跳过，但不能称为模型验证通过。

需要性能测量时，`python tools/profile_pipeline.py --label 本次唯一名称` 使用合成输入；`python tools/compare_matching.py dist/已保存基线.zip` 仅用于可信的本地源码基线。原始结果写 profile_output，不能据合成输入声称识别准确率。

UI 截图使用 `python tools/render_audit.py --output profile_output/ui-check`，真实窗口临时空库；Qt 按钮测试与摄像头硬件链路分开说明。源码用 `python tools/build_source_archive.py` 归档，检查 ZIP 无数据库、人脸、真实清单和权重。不把本地检查通过写成 GitHub Actions 已运行。

交付列出修复、实际检查结果、测量口径和待设备/数据验证项；未经执行的命令只能标为后续步骤。
