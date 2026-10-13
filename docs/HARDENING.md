# v0.4.0 工程审计与交付记录

日期：2026-10-13。范围：可靠性、实验隔离、性能测量、现有桌面流程。技术栈未变。

## 初始状态与保护

目录无 Git 历史。修改前 83 项 pytest 通过、Ruff 检查和格式检查通过；已有多人、轨迹确认、Top-K、CSV 报告、B0–B4 脚本，均复用。

源码基线：`dist/baseline-20261013-before-hardening.zip`（72 文件）。SHA-256：`bc9390469eedc50507677456e768365900d06c0a8f15df7314b96cebe9a6b72f`。这是排除私人图库、模型与环境的源码回退点；原数据未迁移、清理或用于测试。恢复时解压到新目录比较，不覆盖现有数据。

## 问题与修复

| 优先级 | 代码证据 / 复现 | 处理与验证 |
| --- | --- | --- |
| P1 | 删除 DB 行后 unlink 抛 PermissionError，service 无法刷新模板 | 删除意图与 DB 删除同事务；pending_file_deletions 可重试，常见文件占用不再中断模板刷新；故障注入测试 |
| P1 | DB 中图片路径指向样本根外，delete_sample 能删外部文件 | 删除前解析并约束到 samples_dir，越界拒绝且 DB 不变；临时外部文件测试 |
| P1 | 清单改名复制图像可跨 split；实验服务会建空库 | SHA-256 查重并与注册图像比较；实验使用 SQLite mode=ro，缺库立即失败；隔离测试 |
| P1 | 只有人工“冻结”说明，无机器核验 | validation 生成锁；test 必需 --freeze；核验配置、两模型、逻辑库内容/图片、全清单内容、代码和依赖环境；前后双核验 |
| P2 | 特征与库版本分两次读取，跨进程修改可能配错版本 | 同一 SQLite 读事务读取模板与版本；实验入口再核对快照版本 |
| P2 | completed_monotonic 早于日志写入，计时低估 | 完成时间移到日志之后，拆出 log_write 和 lock_wait；日志失败保留识别结果并提示 |
| P2 | Windows 粗粒度 monotonic 时钟可显示端到端小于算法耗时 | 采集、TTL、完成时间统一 perf_counter；视频队列与端到端也使用同一高精度时钟 |
| P2 | 旧模式失败消息没有 generation，仍可清空新结果 | 错误信号携带模式代次；进入模态窗口时推进代次；回归测试 |
| P2 | 已过期 pending 帧仍消耗推理；停止相机阻塞 UI 至多 2 秒 | 推理前 TTL/停止检查；相机非阻塞等待 finished；故障驱动与过期帧测试 |
| P2 | 标准库删除同步阻塞、清空后图片预览可残留 | 后台任务与关闭保护、预览清空、完整性检查和待清理图片重试 |

数据库新增表为向后兼容的 `CREATE TABLE IF NOT EXISTS`。仅正常桌面打开可初始化表；实验只读入口不迁移数据库。突然断电仍可能留下未引用图片，完整性检查报告它们但不自动删除。活跃 DB 的外部修改、损坏数据库恢复、多应用并发写入仍需额外现场评估。

## 验证入口

```powershell
python tools/check_project.py --models
python tools/synthetic_benchmarks.py
python tools/profile_pipeline.py --label unique-run-name
python tools/compare_matching.py dist/baseline-20261013-before-hardening.zip
python tools/render_audit.py --output profile_output/ui-check
python tools/build_source_archive.py
```

第一条组合 Ruff、pytest、模型校验；其他命令分别对应合成 B0–B4、阶段计时、交替 A/B、真实 Qt 截图和白名单打包。测试不打开相机，使用临时库。Matplotlib/Pyparsing 的弃用告警属于现有依赖，未隐藏或改写为成功指标。

本轮组合检查实际结果：105 项 pytest 通过（14 条现有 Matplotlib/Pyparsing 弃用告警），Ruff 检查和 55 个 Python 文件格式检查通过；两个模型 SHA 校验及实际加载通过；项目 skill 校验通过。没有打开真实摄像头，也没有运行远端 CI。

发布归档经独立临时目录解压后执行 `python tools/check_project.py`：104 项通过、1 项真实模型集成测试因包内不含权重而按预期跳过，Ruff 和格式检查通过。归档含 98 个文件，核验没有照片、DB、特征、模型、虚拟环境和原始实验输出；公开图片下载器与去路径汇总报告包含在内。解压测试复用本机虚拟环境，因此验证了源码完整性，不等同于另一台电脑全新安装验证。

详细测量见 [性能报告](PERFORMANCE.md)和[真实公开照片结果](PUBLIC_BENCHMARK.md)，界面证据见 [UI 评审](UI_AUDIT.md)，会话实验计划见 [计划](../refine-logs/EXPERIMENT_PLAN.md)。公开数据测试与日常图库隔离，真实照片/DB/逐图记录均不进入源码包。

## 未验证项

- USB 实际设备选择、首次打开、拔插、长时运行、驱动永久阻塞以及关闭过程。
- 经确认的注册/验证/测试拍摄会话、摄像头准确率/FAR及其会话间波动。公开ORL图片划分已完成，但没有逐图会话标签，不能填补此项。
- 多人快速交叉和遮挡的 ID 切换、首次确认时间、旧姓名残留。
- 1366×768、高 DPI、读屏与完整键盘流程；本轮截图是 Qt 离屏真实控件，非显示器现场测试。
- 人脸质量失败和图片攻击场景；没有专门活体检测，合成输入性能不能用于证明识别安全性。
