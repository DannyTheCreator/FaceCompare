# 实验复现说明

预训练模型的公开成绩不能替代本系统的实际测量。注册集建库，验证集选参数，最终测试集只在配置与库冻结后评价。按拍摄会话隔离；未知人员不进入标准库，验证和测试尽量使用不同的未知身份。

清单参照 `experiments/manifests/example.csv`。路径相对CSV解释，样本ID唯一，会话不为空且不能跨split。已知样本需要标准库中真实的人员ID，未知样本留空。

脚本检查清单编号、图像SHA-256内容重复及注册数据的路径/会话/内容重叠。重新压缩、裁剪或重新编码后的副本仍需人工按会话归属，脚本不能自动核验未知身份。实验入口使用只读SQLite，库必须预先存在，不会自动创建空库或写识别日志。先关闭会写入该库的桌面窗口。

```sh
python experiments/evaluate.py experiments/manifests/local.csv --split validation --calibrate-target-far 0.05 --freeze-output experiments/results/run.freeze.json --output experiments/results/validation
```

在满足有效未知FAR目标的阈值中最大化已知端到端识别率，并列时选较低阈值。必须有有效已知和未知查询；禁止在test上标定。将输出完整精度阈值写回同一配置，不能只抄界面三位小数。freeze文件不覆盖已有版本，再运行：

```sh
python experiments/evaluate.py experiments/manifests/local.csv --split test --freeze experiments/results/run.freeze.json --output experiments/results/test
```

输出 `predictions.csv` 和 `summary.json`，包含计数、模型哈希、预处理版本、库版本及环境。读取失败、无脸、多脸、质量拒绝单列，不算成未知正确拒识。

最终test必须指定--freeze，且不允许--threshold覆盖。锁记录包含生效配置、两模型哈希、库的逻辑内容及注册图片哈希、全部清单标签与文件内容、app/experiments代码和依赖环境；开始及结束时核验。文件缺失会在指纹阶段失败；文件存在但无法解码仍计处理失败。调整质量/检测参数、修改数据/代码、换环境或扩库后要开展新的验证实验，不能直接沿用旧锁。冻结是本地一致性检查，不是防篡改签名。

可在evaluate、benchmark、video_benchmark使用--opencv-threads固定CPU线程数，验证、最终测试及带--freeze的benchmark必须保持一致。变更代码版本后重建实验，不用改写旧锁绕过检查。

| 指标 | 分母 |
| --- | --- |
| 已知正确识别、错认、拒识率 | 全部已知尝试 |
| 未知FAR全尝试口径 | 全部未知尝试 |
| 未知FAR有效查询口径 | 有候选及有限分数的有效未知查询 |
| 未知正确拒识率 | 全部未知尝试 |

无分母时返回null，不记为0。报告同时给出次数和百分比，并按场景分析。

图片对照比较原尺寸/缩小检测与循环/矩阵匹配，每个变体分别预热，遇到不可读图片停止。使用固定阈值比较工作点；若缩放显著改变分布，应另外为各变体用验证集标定，再评价测试集。

```sh
python experiments/benchmark.py experiments/manifests/local.csv --split test --freeze experiments/results/run.freeze.json --repeats 3
python experiments/plot_results.py experiments/results/benchmark/benchmark_summary.csv --output experiments/results/benchmark/latency.png
python experiments/video_benchmark.py data/evaluation/fixed_video.mp4 --repeats 3 --max-seconds 30
```

图片B4组合检测与匹配；视频B4还包括最新帧调度。视频时间从帧进入应用到service结果完成，不包含UI渲染或摄像头曝光前时间。视频未含帧级身份标签，不能评价识别准确率；丢帧缩短排队时间也不等于单次推理加速。

记录CPU、依赖、尺寸、阈值、重复次数和预热。保留原始CSV与失败案例引用；公开展示使用去除姓名和路径的汇总。[ORL公开照片实测](PUBLIC_BENCHMARK.md)提供图片划分成绩与性能数据，真实摄像头会话隔离成绩仍待采集。

公开图片缺少拍摄会话时，清单必须显式 `split_unit=image`，保留如 `orl-session-unavailable` 的真实元数据，评价和图片benchmark必须加 `--allow-image-split`。此模式仍检查图像内容/路径重复、未知身份与注册库关系，并要求test冻结；只放宽不可获得的会话标签隔离，不证明会话泛化。默认省略split_unit仍为严格session模式。使用独立目录和配置，复现入口 `python tools/prepare_orl.py --output data/public/orl/新的目录`，具体划分、质量适配、结果限制及完整命令见上述报告。

图片和视频均输出阶段p50/p95与候选ID。视频逐变体预热后重新打开同一视频，另外记录queue_wait。benchmark不带--freeze时明确为探索性测量；视频无标签实验不输出识别准确率。标准库始终只读。

不具备已核验数据时，运行`python tools/synthetic_benchmarks.py`验证现有脚本；运行`python tools/profile_pipeline.py --label unique-name`测阶段。输入明确标记为合成，输出只写profile_output，不作为最终识别率。使用[实验记录模板](实验记录模板.md)记录运行环境、数据划分、锁定阈值、原始计数和性能结果。

## 图表与多人评估

`python experiments/report.py predictions.csv --threshold 0.42 --output experiments/results/report` 将预测转换为分数分布、1:N工作点曲线和场景汇总，不重新选阈值。汇总不含姓名、人员ID和图片路径，场景标签仍需自行检查。

现有evaluate/benchmark使用单脸查询。多人跟踪需要独立帧级标注视频，记录真实身份和轨迹对应，评价ID切换、入画到确认时间及离开后旧姓名残留。不能用代码测试或单人实验声称多人准确率已经提升。
