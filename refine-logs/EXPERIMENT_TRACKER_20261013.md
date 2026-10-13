# FaceCompare 实验跟踪

日期：2026-10-13。DONE仅针对所列工程验证。

| Run | 阶段 | 输入/目的 | 指标 | 状态 |
| --- | --- | --- | --- | --- |
| R001 | M0 | 临时库/Qt/合成数组 | pytest、Ruff | DONE |
| R002 | M0 | 固定模型SHA+空画面 | 加载与no_face | DONE |
| R003 | M0 | 基线/优化匹配，3 seeds | p50/p95、Top-K一致性 | DONE，见docs/PERFORMANCE.md |
| R004 | M0 | 合成图像/视频/空库，B0–B4 | 算法、队列、吞吐、丢帧 | DONE；不证明准确率 |
| R005 | M1 | 真实注册/会话隔离 | 人数/会话/查重 | TODO，待采集核验 |
| R006 | M2 | validation阈值冻结 | 有效FAR、已知正确率 | TODO |
| R007 | M3 | 冻结test | E1/E2/E4全部指标 | TODO |
| R008 | M3 | 真实输入B0–B4 | p50/p95与输出变化 | TODO |
| R009 | M4 | 多人标注视频 | ID切换、确认延迟 | TODO |
| R010 | M4 | USB/高DPI/现场操作 | 完整手工验收记录 | TODO |

R001–R008及R010为必须项；R009可后续独立推进，但宣称多人准确率前必须完成。

