# guided-mesh-cad

面向 LLM 或操作者辅助定位的实验性网格逆向、原生几何检查和可恢复装配工具。当前版本：**0.3.0.dev23+public.1**；是否发布以同版本 wheel、SHA256 和安装验收记录为准。

输入支持 NPZ 网格、BRep、STEP。曲面分区、功能界面和允许简化的范围由调用者提供；工具不自动识别任意网格的设计意图，也不恢复原始 CAD 特征历史。可处理任意空间方向的多块自由曲面，不要求相同截面。

## 常用命令

```powershell
python -m guided_mesh_cad example workflow --out demo-input
python -m guided_mesh_cad run demo-input/plan.json --out demo-run
python -m guided_mesh_cad status demo-run
python -m guided_mesh_cad resume demo-run
python -m guided_mesh_cad diagnose demo-run --task verify --view errors
python -m guided_mesh_cad diagnose demo-run --task verify --view report --pointer /instances
```

默认返回短摘要，完整报告、原生点证书和日志留在运行目录。旧单步命令保留；需要完整 stdout 报告时显式加 `--full`。状态和已保存诊断不加载 CAD。资源保护用于 `run/resume` 的隔离工作进程，直接 Python API 和旧单步命令由调用者安排资源。

- [运行计划、恢复、按需诊断](docs/workflows.md)
- [本版能力、边界与验证矩阵](docs/dev23.md)
- [既有重建、曲面、材料检查接口](docs/legacy_reference_dev22.md)

## 必须保留的边界

执行完成不等于整机几何、材料、无干涉或连续运动验收。原生实际点给出保守距离上界；未命中才执行边界距离求解。没有覆盖的窄面、用户声明的简化和原始警告保留在报告中。

C0 有效参数域诊断及显式非周期 B-spline 参数域限制不替代完整 BRep 拓扑修复。原生布尔警告不能用有限点检查豁免。

安装到隔离目录验证，不覆盖共用建模环境：

```powershell
python -m pip install --no-deps --target isolated path/to/verified-wheel.whl
```

Python >=3.12；原生测试环境为 CadQuery 2.8.0、OCP 7.9.3.1、VTK 9.6.2。wheel 不自带 CAD 内核，不包含 PDF/U3D 解码器。安装器不修改用户输入。


## 许可证

源码、测试和示例采用 [Apache License 2.0](LICENSE)，版权声明见 [NOTICE](NOTICE)。第三方依赖保留各自许可证。

公开副本使用程序构造的合成输入完成 225 项回归测试，并执行安装后的工作流与重建示例；仅排除 2 项依赖非公开模型的历史测试。此数量不含原版本的独立验收，不能外推到任意输入。
