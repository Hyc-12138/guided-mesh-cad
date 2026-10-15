# guided-mesh-cad

根据明确的曲面分区和参数化方案，将三角网格重建为原生 B-spline 曲面、闭合实体，并导出 STEP / BRep。提供共有边检查、拓扑诊断和网格与原生边界的双向偏差检查。

本工具面向 **LLM 辅助的网格逆向工作流**：由操作者或 LLM 分析几何结构、划分曲面并编制重建方案，工具负责执行几何重建、拓扑检查和偏差验证。熟悉几何与参数化的操作者也可手工编制方案，不强制依赖 LLM。

**本工具不内置 LLM，也不自动调用模型服务。** 如使用 LLM，需自行选择模型及交互方式；LLM 生成的方案仍须由操作者核对，并通过工具的几何与偏差检查。

当前版本：`0.3.0.dev12+public.1`，实验性开发版本。分区、UV 参数化、共有边协调和特征判断由操作者提供；不具备任意网格的全自动逆向能力。输出不包含原设计特征树。

## 安装

需要 Python 3.12 或以上。依赖为 NumPy、SciPy、CadQuery 和 VTK，具体范围见 `pyproject.toml`。本次验证环境为 Windows / Python 3.12；其他平台和 Python 版本未在此发布副本上验证。

在源码目录中执行：

```sh
python -m pip install .
python -m guided_mesh_cad doctor
```

也可以安装随发布提供的 wheel：

```sh
python -m pip install guided_mesh_cad-0.3.0.dev12+public.1-py3-none-any.whl
```

wheel 包含本项目的 Python 代码与示例生成器。原生 CAD 依赖由 pip 单独安装，未内置在 wheel 中。

## 快速开始

所有示例都通过数学公式现场生成，不需要外部模型文件。使用新的输出目录：

```sh
python -m guided_mesh_cad example loop --out demo-input
python -m guided_mesh_cad reconstruct demo-input/plan.json --out demo-output
python -m guided_mesh_cad diagnose-step demo-output/model.step --report topology.json
python -m guided_mesh_cad validate demo-input/source.npz demo-output/model.step --report check.json --tolerance-mm 0.01
```

`example freeform` 生成六块不对称曲面和斜向贯穿孔的合成示例；`example loop` 生成空间起伏闭环的十六块曲面。示例用于检查数值流程，不证明未知模型的自动分区能力。

CLI 退出码：`0` 为当前命令通过，`2` 为检查完成但不满足条件，`1` 为输入或运行错误。采样检查通过仅适用于对应输入和检查范围。

## 输入与重建

输入 `.npz` 包含毫米制 `vertices`（N×3 坐标）及 `faces`（M×3 整数三角形索引）。plan JSON 绑定输入文件 SHA-256，并声明范围、曲面块和验证参数。可运行的完整 plan 由示例命令生成。

- `indexed_grid`：提供原始顶点索引网格。
- `uv_chart`：提供原始三角网格的顶点、三角形及显式 UV 参数域，按实际三角形重采样。
- `construction_mesh`：可选辅助构造网格，需单独绑定哈希及依据；最终偏差仍对完整输入网格检查。
- `cylinder_cut`：显式圆柱切除，要求方向、位置、尺寸和来源说明。

每个曲面块需要四边参数域。不能表示的区域需要先拆分；开口、错误拓扑或超差会失败。缝合容差受到验证阈值约束，不能靠放宽容差替代正确边界。

主要输出：`model.step`、`model.brep`、`manifest.json`、`verification.json`、`fitted_grids.npz` 和 plan 副本。报告记录输入输出哈希、范围及检查结果。运行错误保存 `failure.json`；重建拒绝覆盖非空输出目录。

## Python API

```python
from guided_mesh_cad import (
    resample_closed_contours_by_anchors,
    simplify_corresponding_polylines,
    sample_native_edge_bounded,
    validate_native_boundary,
    validate_native_boundary_uv_witness,
    validate_native_boundary_low_memory,
)
```

- 轮廓锚点重采样保持调用方声明的跨站对应关系，不自动判断锚点语义。
- 对应折线简化保留指定特征点，在相同参数处约束输入分段线性折线的偏差；不保证后续高阶插值的误差。
- 原生曲线采样使用导数控制界细分，支持有限裁剪直线、圆、椭圆、Bezier 及 B-spline；预算不足或不支持的曲线会明确失败。
- `validate_native_boundary` 对实际裁剪 CAD 边界检查，包含全部源顶点、三角形中心、随机点及原生边界见证点。
- `validate_native_boundary_uv_witness` 使用真实原生曲面点提供距离上界，无法接受提示点时回退到原生边界投影。
- `validate_native_boundary_low_memory` 将完整采样数组存入磁盘映射文件，保持检查范围；需要新的 `diagnostics_dir`。

`guided_mesh_cad.cad` 还提供 `canonical_shared_edge`、`validate_shared_boundaries`、`face_from_surface_and_wires` 和 `sew_solid_with_report`。共有边需复用同一原生 Edge，几何重合的独立边不能替代拓扑共享。

## 限制

有限采样、网格化精度和浮点导数界均不是全局 Hausdorff 误差的严格证书。不自动保证缺失特征恢复、输入完整性、制造公差、装配配合、连续运动或跨面 G1/G2 连续性。重建流程面向单闭壳实体，不自动恢复多实体装配或封闭内腔。

磁盘映射减少 Python 数组内存占用，但不限制原生 CAD 内核的内存。大型曲面仍需外部资源监测。长时间原生调用可通过 `progress` 回调观察；当前执行建议避免周期性强制线程堆栈采样，其与原生计算的相互影响尚未完全查明。

运行生成的报告和诊断缓存包含调用方输入路径及几何数据，应留在本地输出目录。

## 开发与验证

```sh
python -m pip install .
python -m unittest discover -s tests -v
python -m pip install build
python -m build
```

测试使用程序生成的几何及故意构造的错误输入，覆盖共有边、壳闭合、容差拒绝、曲线采样、轮廓处理和边界验证。构建产物位于 `dist/`。发布验证范围见 `VALIDATION.md`。

项目只调用依赖库的公开接口，不随源码或 wheel 重新分发依赖库二进制。
