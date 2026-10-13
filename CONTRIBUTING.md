# Contributing

欢迎提交可复现的问题和改进。修改围绕具体行为，附带能证明行为的测试或实验记录。

```sh
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest -m "not integration" -q
python tools/check_project.py
```

需要真实模型测试时先下载模型，再运行 `python -m pytest -m integration -q`。测试使用临时数据，不读取开发者的真实 `data/gallery.db`。

UI负责展示，算法通过service调用，持久化交给repository，实验复用相同识别链路。耗时任务使用线程，销毁窗口前确保线程退出。配置增加兼容默认值并验证范围；影响特征空间时更新预处理版本。

PR说明问题、结果及验证。算法修改记录数据划分、模型、阈值和性能口径；不使用伪造指标或注册照片上的表现证明泛化。

提交前确认实验test依旧强制冻结、测试数据库在临时目录、QThread关闭保护有效。截图工具只使用临时空库；profile_output及freeze文件保持本地。

提交前查看 `git diff --cached --stat`。仓库不收录个人照片、身份信息、特征向量、数据库、真实实验清单和模型权重。截图与日志移除姓名、本机路径；Issue提供运行环境、步骤、预期和实际结果。
