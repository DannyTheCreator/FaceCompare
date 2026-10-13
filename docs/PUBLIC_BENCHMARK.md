# AT&T/ORL 公开照片实测

日期：2026-10-13。本报告是实际执行结果，使用现有 evaluate、benchmark、report 和 YuNet/SFace，没有训练模型。数据只保存于本地独立 `data/public/orl/`，没有使用日常桌面图库；源码包不包含照片、特征或数据库。

## 来源与实验边界

[剑桥大学托管的 AT&T/ORL 官方说明](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)：40 人、每人 10 张、92×112 灰度 PGM。按原站要求致谢 **AT&T Laboratories Cambridge**。下载固定官方 ZIP，SHA-256 为 `509d763bd2276aa054d260d7f1000e8be212738134f2f83d8ff13bc2adf1c8ec`。数据不适用本项目的 MIT 许可证；本次仅下载供本地实验，不再分发原图。

原站说明部分照片拍于不同时间，但文件没有逐张会话标签。因此全部保留 `session_id=orl-session-unavailable`，清单明确 `split_unit=image`；评价入口必须显式传 `--allow-image-split`。不编造拍摄会话，不将此结果称为会话隔离或摄像头泛化实验。普通清单仍默认 session 模式。

固定划分在最终测试前完成，不按测试表现删样本：

| 用途 | 身份与照片编号 | 数量 |
| --- | --- | --- |
| 注册 | s1–s20，各 1–5 | 100 |
| 已知 validation | s1–s20，各 6–7 | 40 |
| 未知 validation | s21–s30，各 1–10 | 100 |
| 已知 test | s1–s20，各 8–10 | 60 |
| 未知 test | s31–s40，各 1–10 | 100 |

未知验证和测试身份不重叠。所有输入统一 4 倍 INTER_CUBIC 放大为 368×448，不使用人工人脸框。首次默认质量门限的注册试跑 run-01 在 s1 的 5 张注册图全部得到 too_blurry 后停止；这些图放大后的 Laplacian 方差约 4.62–8.02，低于摄像头默认门限 40。仅依据注册图诊断，为独立 run-02 关闭模糊门限（min_blur_variance=0），保留亮度、人脸面积和检测分数门限；这不是默认摄像头配置的成绩。独立配置 minimum_samples=1，实际 20 人均成功注册 5 张，100/100 成功。应用默认配置没有改变。

## 冻结与结果

validation 目标为有效未知 FAR≤5%，选择阈值 `0.39230034814605713`，随后写入独立配置、冻结标准库与全清单，并执行一次最终 test。后续性能重复沿用此阈值，没有在 test 调参。锁覆盖模型、代码、环境、配置、全清单和逻辑标准库；原始日志和锁位于 `data/public/orl/run-02/`。

| 指标 | validation | test |
| --- | --- | --- |
| 已知正确 / 全部已知 | 40/40（100%） | 60/60（100%） |
| 未知正确拒识 / 全部未知 | 95/100（95%） | 94/100（94%） |
| 未知误接受 / 有效未知 | 5/100（5%） | 0/94（0%） |
| 未知误接受 / 全部未知 | 5/100（5%） | 0/100（0%） |
| 未检出人脸 | 0 | 6（全部为未知输入） |

6 次未检出是处理失败，不计作正确拒识。样本量小、同一人的照片相关，未核验预训练数据重叠，不能据 0 次误接受宣称部署 FAR 为 0，也不能声称整体准确率 100%。阈值和图库规模绑定；这里是 20 人标准库。

## 同输入性能与输出变化

Windows build 26200，Python 3.11.7，OpenCV 4.12.0，NumPy 2.2.6，AMD64 Family 25 Model 97 Stepping 2，OpenCV 1 线程。模型哈希见 THIRD_PARTY_NOTICES.md。每变体独立预热 1 次，同一 160 张 test 图片重复 3 次；按固定变体顺序运行，没有随机化顺序。下表为每变体 480 次时延的合并分位数；特征/匹配统计只包含成功进入该阶段的样本。

| 变体 | 检测 p50/p95 ms | 特征 p50/p95 ms | 匹配 p50/p95 ms | 算法总计 p50/p95 ms |
| --- | --- | --- | --- | --- |
| B0 原尺寸、循环 | 9.242/9.647 | 25.592/26.272 | 0.087/0.095 | 35.505/36.511 |
| B1 缩小检测、循环 | 2.592/3.637 | 25.705/30.595 | 0.088/0.132 | 29.004/35.181 |
| B2 原尺寸、矩阵 | 9.324/9.560 | 25.699/25.982 | 0.076/0.084 | 35.669/36.227 |
| B4 缩小检测、矩阵 | 2.611/2.758 | 25.797/26.767 | 0.075/0.086 | 29.076/30.350 |

B0→B4 算法 p50 下降 18.1%，p95 下降 16.9%。这是已有检测缩放/矩阵能力的消融，不是本轮新增优化的前后比较；新增 Top-K 优化的交替 A/B 见 [PERFORMANCE](PERFORMANCE.md)。真实小库主要瓶颈仍为 SFace，不能把微秒级匹配改善写成整体显著提速。3 次运行算法中位数分别为：B0 35.426/35.637/35.510 ms；B4 29.042/29.187/29.053 ms。B1 尾延迟存在波动，不按它与 B4 的差值推断矩阵匹配收益。

- B0↔B2、B1↔B4：各 480 次配对中候选身份和状态变化均为 0；最大相似度差 `1.1920928955078125e-7`（浮点归约差）。
- B0↔B4：每轮 160 图中 31 图候选字段变化、10 图状态变化；5 图由 no_face→unknown，另 5 图反向变化，最大有效分数差约 0.079155。全部已知身份仍正确，未知误接受仍为 0，但检测覆盖的样本不完全相同，不能声称缩放输出不变。
- 图片串行实验没有预览和队列，不能把 `1000/p50` 写成桌面 FPS。B3 最新帧队列的实测目前仅有 [合成视频结果](PERFORMANCE.md)，真实摄像头/有脸视频时延和丢帧仍待验证。

公开的表格为去路径汇总，逐图 CSV、原图和 DB 留在本地。源码版本指纹 `1b5d8e07d1e82dfe30cf05a50130013aaa63a7e4a0c5d998dc6c773cd89a884c`；标准库指纹 `06d1cc48e46aa9457b7caaa40b421cb22d1fe2ba35a77b54d17a0a1437f52f8d`；清单内容指纹 `b2bccb6dea3ba9cdd340351aa65f7cf842ce88ac481f751a224cfe4dbcae695e`。这些是本次记录，重建图库后应生成新锁，不复制旧锁。

## 可复现命令

先安装项目依赖并下载通过校验的两个模型。PowerShell 从仓库根目录运行，选择不存在的目录（不要覆盖 run-02）：

```powershell
python tools/prepare_orl.py --output data/public/orl/run-03
$run = 'data/public/orl/run-03'
python experiments/evaluate.py "$run/manifest.csv" --config "$run/config.json" --split validation --allow-image-split --opencv-threads 1 --calibrate-target-far 0.05 --freeze-output "$run/run.freeze.json" --output "$run/validation"
```

将这次锁中的完整精度阈值写入这次配置（不改根目录 config.json）：

```powershell
python -c "import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); c=json.loads((p/'config.json').read_text()); f=json.loads((p/'run.freeze.json').read_text()); c['matching']['cosine_threshold']=f['settings']['matching']['cosine_threshold']; (p/'config.json').write_text(json.dumps(c,indent=2),encoding='utf-8')" $run
python experiments/evaluate.py "$run/manifest.csv" --config "$run/config.json" --split test --allow-image-split --opencv-threads 1 --freeze "$run/run.freeze.json" --output "$run/test"
python experiments/benchmark.py "$run/manifest.csv" --config "$run/config.json" --split test --allow-image-split --opencv-threads 1 --freeze "$run/run.freeze.json" --repeats 3 --output "$run/benchmark"
$threshold = (Get-Content "$run/run.freeze.json" -Raw | ConvertFrom-Json).settings.matching.cosine_threshold
python experiments/report.py "$run/test/predictions.csv" --threshold $threshold --output "$run/report"
```

查看 provenance.json、validation/summary.json、test/summary.json、benchmark/benchmark_runs.csv 和 benchmark_summary.csv。图像下载有大小与 SHA 校验；默认 pytest 不访问网络。数据用于本地实验，不随仓库分发。
