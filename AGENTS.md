# FaceCompare 项目约定

本目录是仓库根目录。保留 PySide6 + OpenCV YuNet/SFace；先读 README、docs/ARCHITECTURE.md、docs/EXPERIMENTS.md。修改前检查 Git 状态；没有 Git 时用 tools/build_source_archive.py --output dist/独立基线名称.zip 保存源码，不覆盖旧基线。

- UI 只展示和派发；推理走 service，持久化走 repository。耗时操作用工作线程，禁止销毁运行中的 QThread。
- 时间戳统一用 time.perf_counter()，采集到完成使用同一时钟；过滤 stream、gallery version、mode generation 和 TTL。模板内容与版本在同一数据库读事务中获取。
- 特征空间变动要更新 preprocess_version 并重建兼容标准库；不迁移或覆盖用户图库来让测试通过。
- 测试只用临时数据库、合成图像或明确授权的独立数据。不得读取开发者真实图库来跑默认测试；不得上传照片、特征、数据库或私人清单。
- 实验库只读。默认注册/验证/测试按会话隔离；无会话标签的公开基准必须明确 split_unit=image 并显式 --allow-image-split，不声称会话隔离。阈值仅在 validation 标定，test 强制 --freeze。不要用最终测试选阈值。
- 改动后运行 python tools/check_project.py --models（模型缺失时先省略 --models）；真实模型集成用空画面，不开启摄像头。只重跑相关失败或被修改的检查。
- 项目专用复核流程见 [facecompare-check](skills/facecompare-check/SKILL.md)，保持文档命令与代码一致。源码发布用白名单归档；发布、推送需用户明确要求。
