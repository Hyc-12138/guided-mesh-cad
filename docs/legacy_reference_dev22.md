# guided-mesh-cad：有来源约束的逆向实验工具

以下为既有接口的技术参考；当前版本及使用边界见 README。安装成功或局部检查不能替代完整来源验证。

面向 LLM / 操作者指导的网格逆向实验工具。可将**任意空间方向的多块四边自由曲面**重建为 B-spline 原生面，严格检查缝合，构成单一实体，叠加有明确依据的斜向圆柱孔，并导出 STEP / BRep。此路线不要求等截面、公共轴或规则旋转外形。

**当前是实验版，不是任意网格全自动逆向器。** 分区、UV 参数化、共有边界协调和特征判断仍由 LLM / 操作者提供。每个曲面块必须有可用的四边参数域；任意拓扑要先拆分为可表示的区域。不能表示的区域、开口、错误实体或超差会失败，不会悄悄退化成三角面 STEP。

## 0.3.0.dev16 候选：多边自由曲面的分区与控制网

本节描述新增接口；是否已发布、安装验证和独立验收范围仍以版本对应的验收记录为准。

`prepare_source_disk_quads(vertices, triangles, boundary_cycle, anchor_indices, samples=65, source_binding=...)`
接收一块完整的、有方向且流形的三角网格圆盘域，以及调用者指定的完整边界环和至少三个语义角点。它按正权调和映射建立凸多边形参数域，再为每个语义角点生成一块四边网格。不要求公共轴、相同截面或规则外形。返回每个采样点对应的**原三角面索引与重心坐标**，保留全部输入三角面；不自动焊点、重三角化或填孔。带孔、断开、非流形或无法建立无翻转二维参数域的输入明确拒绝，须先由操作者分区。

`boundary_cycle` 使用原顶点索引，末尾不重复起点；`anchor_indices` 是这个边界序列中的位置，按一圈顺序排列。返回字典包含 `grids`、`source_uv`、`quad_source_uv`、`lineage`、`boundary_chains` 和 JSON 格式的 `report`。`source_orientation_in_chart` 表明曲面参数方向与原三角方向的关系；生成原生面时须继承该方向。

`make_control_face(control_grid, tolerance=1e-7)` 将至少 4×4 的有限三维控制网作为三次、内部 C2 的正基函数 B 样条**控制点**。这是显式近似接口，不保证穿过控制点；原 `interpolate_face` 不变。控制凸包不能代替源误差、法向、局部正则性或全局自交检查。调用者仍须协调不同源块的共有边界、核真实原生边归属、缝合和完整实体，并对原始来源及所有原生面做实际双向检查。

```python
atlas = prepare_source_disk_quads(
    vertices, triangles, boundary_cycle, anchor_indices,
    samples=65, source_binding={"source_sha256": source_hash, "selector": source_scope})
faces = [make_control_face(grid) for grid in atlas["grids"]]
```

以上仅生成候选曲面，不是完整模型或验收代码。相邻源块的外边界没有在此接口中自动合并；源码分区通过、原生 Face 有效或控制点密度增加，都不能直接写成整体逆向通过。

## 安装及运行

测试环境：Windows、Python 3.12、CadQuery 2.8.0 / OCP 7.9.3.1、VTK 9.6.2。

```powershell
python -m pip install path/to/independently-accepted-wheel.whl
python -m guided_mesh_cad doctor
python -m guided_mesh_cad example loop --out loop-input
python -m guided_mesh_cad reconstruct loop-input/plan.json --out loop-output
python -m guided_mesh_cad reconstruct path/to/plan.json --out path/to/new-run
python -m guided_mesh_cad validate source.npz new-run/model.step --report check.json --tolerance-mm 0.01
python -m guided_mesh_cad diagnose-step candidate.step --report topology.json
python -m guided_mesh_cad diagnose-native candidate.brep --report topology-brep.json
```

wheel 包含 Python 实现，平台原生 CAD 依赖由 pip 单独解析；不是自带内核的离线安装包。已有正确依赖时可用 `pip install --no-deps --target isolated-directory wheel-path`，再将该目录加到 `PYTHONPATH`。不应覆盖共用运行环境。

退出码：0 = 有限采样检查通过；2 = 采样偏差不通过但保留结果；1 = 输入/建模/运行错误。通过不代表制造精度、产品结构确认或正式工程验收。

## 输入及工作流程

1. 输入 `npz`：`vertices` 为毫米制 N×3 坐标，`faces` 为 M×3 整数三角面索引。输入必须包含待验证的全部表面。原始网格的孔、凹陷和倒扣不能在验证时省略。
2. 人工/LLM 判断特征和曲面分区，提供 plan。支持 `indexed_grid` 原始顶点索引网格，以及 `uv_chart` 的原始三角网格+显式 UV，后者按实际三角面重采样，不填补 UV 域中的孔或域外区域。UV 折叠在采样位置检测；当前未证明连续域全局无折叠。
3. 以独立曲面块插值，无轴向限制。边界相合时缝合实体；本版要求一个闭合 shell，不支持多个独立 shell 表示封闭内腔。斜孔使用显式 `cylinder_cut`，其它解析特征暂未实现。
4. STEP 全部根节点回读，检查有效性/连通性/体积，再将**所有表面**和源网格做双向采样距离检查；最差若干点额外测原生 BRep 边界距离。裁剪曲面网格化尊重孔洞，不能取整张 UV 包围矩形替代。

plan 必须绑定原网格 SHA256。字段和完整可运行样例由 `examples/make_freeform_example.py` 生成：

```json
{
  "schema_version": 1,
  "units": "mm",
  "scope": "明确写出整件还是局部范围",
  "mesh": "source.npz",
  "mesh_sha256": "实际文件哈希",
  "patches": [
    {"id": "patch1", "kind": "indexed_grid", "indices": [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]}
  ],
  "verification": {"tolerance_mm": 0.01, "samples": 3000}
}
```

上面只展示字段，不是完整实体；至少提供四个能形成闭壳的曲面块。索引网格行列均需 4–256 个点。`uv_chart` 使用 `vertex_ids`、对应 `uv`、源 `triangle_ids`、`grid_shape: [25,25]` 和可选 `uv_bounds: [[0,1],[0,1]]`。这是显式参数化，不是自动分区。

如果需要推断孔之前的基底曲面，可以另提供 `construction_mesh`、`construction_mesh_sha256`、`construction_evidence`。曲面块从该辅助网格读取，**最终偏差仍对原始 mesh 全部表面检查**。辅助网格的推断过程必须标明，不能把推断出的点说成原始测点。

每个解析孔必须给出来源 `evidence`、`origin`、`direction`、`radius`、`length`，全部长度毫米。缝合容差默认 0.000001 mm，不能超过 0.001 mm 或验证阈值的十分之一。不通过不能靠增大缝合容差强行闭合。

## 复杂样例

```powershell
python examples/make_freeform_example.py --out demo-input
python -m guided_mesh_cad example freeform --out another-demo-input
python -m guided_mesh_cad reconstruct demo-input/plan.json --out demo-output
```

样例为三维不对称变形的六面体，带偏心凹陷和斜向贯穿孔。25×25 的构造网格用于重建，49×49 的较密独立采样参考曲面网格用于比较。参考体和测试体共用内核，因此此样例只验证数值路线，不能证明真实任意网格的自动分区。独立测试使用额外生成的不同网格及坏例。

另一个样例由 `examples/make_loop_example.py --out loop-input` 生成：空间起伏、不等厚、表面不对称的完整闭环，带通孔拓扑和倒扣，使用 16 个曲面块。它直接比较解析公式生成的密集原网格与由稀疏原始顶点重建的实体，不采用轴向截面或人工端盖。

## 输出与边界

- `model.step` / `model.brep`：原生曲面实体，带有效性结果；几何可继续进行 CAD 布尔运算。没有恢复原设计特征树。
- `manifest.json`：版本、输入/输出哈希、范围、实体数、曲面类型、体积及最终检查状态。
- `verification.json`：双向采样数、最大/P95/RMS 偏差、原生边界复核。包含所有源顶点及面积随机样点。有限采样和网格化请求精度都不是全局 Hausdorff 误差保证。
- `fitted_grids.npz` / `plan.json`：可追溯参数证据。开始运行后的异常另写 `failure.json`，不会标成通过。非空输出目录会在运行前拒绝，保持原内容不变。

当前未自动解决：分区、参数化、噪声处理、缺失面修复、精确尖棱识别、跨面 G1/G2 连续性、闭合内部空腔、多实体装配、任意完整产品逆向、投影出图 与连续运动验收。复杂曲面的计算成本可能较高。

## 0.2 共有边与拓扑诊断 API

`canonical_shared_edge(points, tolerance, source_binding)` 从冻结的来源点只生成一次原生 Edge，不平均、不吸附、不移动来源坐标。生产任务的 `source_binding` 应含源文件 SHA-256 和点集选择说明；它原样进入报告。相邻面应复用该 Edge 或其 `reverse()` 方向。`validate_shared_boundaries()` 要求每一对边显式声明 `alignment="same"` 或 `"reverse"`，并强制 `same_topods_edge=true`；仅几何重合但各自生成的两条边始终失败。

```python
from guided_mesh_cad.cad import canonical_shared_edge, validate_shared_boundaries

binding = {"source_sha256": source_sha256, "selector": "patch A v=1 / patch B v=0"}
edge, source_report = canonical_shared_edge(source_points, tolerance=1e-7,
                                             source_binding=binding)
seams = [{
    "id": "upper/flange",
    "edge_a": edge,
    "edge_b": edge.reverse(),
    "alignment": "reverse",
    "samples": 17,
    "max_gap_mm": 1e-7,
    "source_binding": binding,
}]
seam_report = validate_shared_boundaries(seams, default_tolerance=1e-7)
```

`face_from_surface_and_wires(surface, outer_wire, inner_wires)` 用调用方提供的原生 Wire 裁剪曲面，供相邻面复用同一 Edge；它不投影或修补来源曲线。`sew_solid_with_report(faces, tolerance, shared_boundaries)` 还会确认每条已声明共边恰由两个待缝合输入面持有，避免用与面网无关的重合曲线生成假阳性报告。成功时返回 `(solid, report)`；失败时抛出 `PatchTopologyError`，异常的 `report` 保留共有边实测及所属面、free/multiple edge 数、shell 数、逐面 `BRepCheck` 状态和退化边。旧 `sew_solid()` 保持兼容，只返回实体。

`diagnose-step` 可对有效或无效 STEP 做只读检查；`diagnose-native` 另接受 STEP 或 BRep。退出码 0 表示一个有效闭合实体，2 表示检查完成但不满足条件，1 表示读取或运行失败。单一 Solid 数不再被当作有效几何证明。该诊断不会自动 heal，也不会把三角网格重新包装成 STEP。

本实现直接调用 CadQuery/Open CASCADE，不包含第三方逆向项目的源代码。

## 0.3 开发中：显式张量参数与阶数

`interpolate_face(grid, tolerance, u_parameters=None, v_parameters=None, u_degree=None, v_degree=None)` 在未给参数和阶数时保持旧 `Approx_IsoParametric` 路径。给出任一参数或阶数后，使用 SciPy 张量插值构造原生 `Geom_BSplineSurface`；缺失参数方向使用归一数据索引。参数必须有限、长度匹配且严格递增，阶数必须是 `1..min(3, 该方向点数-1)`。构造后会在全部输入节点复核，超出构造容差即失败。

显式参数只定义样本在参数域中的位置，阶数只控制插值函数；二者都不证明节点之间的曲面贴近来源。独立全周弧长归一可能使不同截面的角点和边界失去对应，造成区间过冲。此类输入须按来源角点和边界分 chart、固定语义边界，并以完整来源的双向最大偏差验收；不能靠阶数、节点误差、轴向单调、体积或 p95 宣称通过。

`resample_closed_contours_by_anchors(contours, anchor_indices, samples_per_segment, tolerance=1e-9, expected_orientation="ccw", source_binding=None)` 按调用方逐站给出的同语义锚点拆分闭合轮廓，不自动识别角点或槽边。输入轮廓必须显式闭合，可在各站具有不同点数；方向检查使用坐标前两维的有符号面积。锚点必须沿输入方向恰好绕行一周，且各站数量和语义次序一致。每段独立按折线弧长重采样，样本数包含两个端点，端点直接复制来源坐标。返回的每个 chart 形状为 `(段内样本数, 站数, 坐标维数)`，并附闭合误差、方向、锚点、段长、端点误差及来源绑定报告。该函数拒绝开环、反向、重复或越界锚点、零长段和语义数量不一致；它不判断调用方锚点语义是否选对，也不代替曲面偏差验收。

技术依据：[Open CASCADE 曲面插值文档](https://occt3d.com/dev/doc/refman/html/class_geom_a_p_i___points_to_b_spline_surface.html)。依赖库遵守各自许可证；不在 wheel 中重新分发其二进制。

## 0.3.dev1：多站对应折线的有界简化

`simplify_corresponding_polylines(grid, parameters, max_deviation, required_indices=(), source_binding=None)` 接收 `(节点数, 站位数, 2或3维)` 的坐标和严格递增的公共参数，返回精简坐标、保留参数及检查记录。保留两端和全部指定特征节点，保留点的坐标不移动。它在所有站位共同选择节点，避免各站各自简化后丢失对应关系；同一公共边两侧的全部站位应先合并输入，再使用同一结果。

误差是**相同参数处的欧氏距离**，检查每个原始折线节点。原折线与简化折线之差在每个原始参数区间内为仿射函数，其范数不超过两端值，因此该检查约束所提供的整条分段线性折线。仅在两模型均采用相同线性站间插值时，才继承站间界。记录浮点实测值，不宣称区间算术证明。

```python
from guided_mesh_cad import simplify_corresponding_polylines

reduced_grid, reduced_u, report = simplify_corresponding_polylines(
    source_grid, source_u, max_deviation=0.00002,
    required_indices=source_feature_indices,
    source_binding={"source_sha256": digest, "selector": "source chart and all stations"},
)
```

示例误差值是调用示意，不是自动适用的制造公差。该函数不识别特征，不证明拓扑，不约束随后高阶样条的过冲，也不代替原始网格与最终实体的完整双向检查。用于降低输入冗余，不能据此删去未声明但真实存在的孔、锐角或材料区域。

不同版本分别验证安装包；新增功能不能继承旧版本的测试结论。
# Failure-location evidence (0.3.0.dev2 candidate)

`pipeline.validate(...)` also records the eight worst sampled coordinates in each direction. Optional `diagnostics_dir=...` saves the already measured native tessellation with its SHA256 and projects the eight worst reverse samples onto the actual native boundary. This avoids repeating a slow CAD mesher just to locate a failure. Use a fresh empty diagnostics directory; source meshes and frozen reports are never overwritten.

The supplemental projection does not lower the acceptance threshold, remove a failed sample, or replace the original finite-sample maximum. The cached triangles are inspection evidence, not an editable reconstructed model. Large CAD models can produce large optional cache files. This development candidate needs independent installed-wheel validation before release.

# Actual trimmed-boundary metric (0.3.0.dev3 candidate)

`guided_mesh_cad.validate_native_boundary(source_npz, native_step, fresh_report, samples=100000, diagnostics_dir=fresh_empty_dir)` compares the supplied mm source with actual trimmed CAD face and edge points. It covers every source vertex and triangle centre, every native face mesh-UV node and triangle centre, random samples in both directions, and native edge/vertex witnesses. Out-of-trim UV locations map to real trim edges and are counted. Inspection triangles provide search hints and UV sample locations; their approximate 3-D positions do not stand in for CAD geometry.

The existing mesh-based `pipeline.validate` is unchanged. Keep its failures as separate evidence. The new method requires a fresh report path, records its distinct metric and full coverage, and remains a finite-sample check. It does not certify a global Hausdorff bound, original CAD history, source completeness, manufacturing intent, assembly fit, or motion. This candidate needs installed-wheel independent tests before release.

# Native integrity guard (0.3.0.dev4 candidate)

All formal solid readers and validators now reject native vertex, edge or face tolerances above 0.01 mm and topology outside the complete solid. The reports expose these values separately from Open CASCADE validity. Diagnostic unchecked readers remain available and preserve the offending geometry. This fixes an input guard omission; it does not heal geometry, clamp tolerances, alter the boundary metric, or replace frozen earlier results. Previously accepted inputs with excessive native tolerances or detached topology will now fail explicitly. Installed-wheel independent validation remains required.

`0.3.0.dev5` also applies that same formal solid gate to the exit status of `diagnose-step` and `diagnose-native`, including positive material volume. A successfully read but failing input still gets its complete diagnostic report and exit status 2; it is not a read error. Open CASCADE `valid` remains a separate field. Earlier frozen reports and wheel installations are retained.

## Native curve sampling with a full-interval derivative bound (0.3.0.dev6)

`sample_native_edge_bounded(edge, max_deviation_mm=0.0001, *, max_segments=200000, max_depth=30, source_binding=None)` returns native curve points, increasing native parameters and an interval report. It supports finite trimmed lines, circles, ellipses, Bezier curves and rational/nonrational B-splines, including located and periodic B-splines. It preserves every knot span and the native edge without healing, moving or fitting it. Unsupported or degenerate curves and insufficient subdivision budgets fail explicitly.

Each interval uses the linear interpolation error bound `sup(norm(C''))/8` in its local unit parameter. Polynomial derivative control hulls, positive-weight rational quotient bounds and exact native Bezier decomposition provide full-interval bounds. An S-shaped cubic with collinear endpoints and midpoint therefore cannot pass merely by midpoint aliasing. Reports include the floating numerical budget, native tolerances, endpoint vertex offsets, decomposition endpoint checks, segment bounds and optional frozen input binding. Numerical bounds use floating arithmetic; they are not interval arithmetic certificates.

The bound covers the actual trimmed geometric curve and its same-parameter polyline in both directions. It does not cover topological vertex offsets, pcurves, source-mesh distance, a later loft/interpolated surface, whole solids or assembly fit. Requested precision below the numerical budget, nonpositive rational weights, overflow and unsupported geometry are rejected. This capability needs installed-wheel independent validation before being used as accepted tooling; the prior dev5 installation remains available.

## Shared native edge ownership indexing (0.3.0.dev7)

Declared seam validation now builds an Open CASCADE `IndexedMapOfShape` once to count input face owners and reject duplicate declarations. Its identity semantics match `TopoDS_Shape.IsSame`: orientation is ignored for identity, location is included, and coincident independent edges remain distinct. The separate orientation, sampled geometry, source binding, input-face and formal-solid checks remain unchanged. Repeated input face positions retain their original owner-list behavior. This removes repeated full face-edge enumeration per declared seam without omitting faces or changing the required two input owners. Build inputs are fresh source copies and wheel/installed Python bytes are checked against frozen source, avoiding timestamp-based reuse of stale build files. Prior installations and failures remain preserved; installed-wheel independent validation is required for this candidate.

## Explicit native UV point witnesses (0.3.0.dev8 candidate)

`validate_native_boundary_uv_witness(source_npz, native_step, fresh_report, samples=100000, diagnostics_dir=fresh_empty_dir)` is a distinct conservative metric (`actual_trimmed_native_boundary_uv_witness`). It retains every original source vertex/triangle centre and all original random samples, every native face UV node/triangle centre/random sample and all native edge/vertex witnesses. Geometry, tolerance, material, ownership and trim gates are unchanged. The existing `validate_native_boundary` entry point defaults to the original full projector path and retains its original metric; its frozen reports are not rewritten.

For a source point, the nearest inspection triangle only supplies a barycentric UV seed. The seed is evaluated on the actual native surface, strictly classified IN its real trimmed face and measured in 3-D against that source point. Only an actual-point distance at or below the unchanged limit may be accepted as a witness. An ON/outside, unusable, failing or over-limit seed goes through the original global surface projection and complete native boundary fallback. No inspection triangle coordinate is accepted as CAD geometry. Diagnostics retain each witness flag, actual native UV, face ID, projected native coordinate and measured distance for audit.

The source metric is an actual-point distance **upper bound**, not a claim to the globally nearest point. It may be more conservative than the original projector's numerical value. This is the reason for the separate entry point/method name. The complete reverse metric remains unchanged; no source point, native face, hole or failing hint is omitted. Finite sampling does not prove global Hausdorff distance, manufacturing intent, source completeness, motion or assembly fit. Installed-wheel independent tests are required before accepting this candidate.


## Computed shell closure (0.3.0.dev9 candidate)

Shell closure now requires BRep_Tool.IsClosed and BRepCheck_Shell.Closed/Orientation NoError. The cached TopoDS Closed flag is retained in diagnostics and never changed or used as sole evidence. Public sewing and formal solid reports consume the same computed result. Open shells with a true cached flag are rejected; genuinely closed shells with a false cached flag can proceed to unchanged validity, positive volume, ownership and native tolerance guards. No geometry repair, tolerance clamp or source omission is performed. This does not assert source distance acceptance or whole-case completion. Installed-wheel independent verification is required.

## Explicit low-memory actual-boundary validation (0.3.0.dev10 candidate)

`validate_native_boundary_low_memory(source_npz, native_step, fresh_report, diagnostics_dir=fresh_directory, samples=100000, mesh_parallel=False)` keeps the whole native mesher at the same absolute tolerance/10 and angular .05 settings. It writes every face UV node and triangle to NumPy disk mappings, computes dense sampling arrays in bounded arithmetic blocks, and retains every native node, triangle centre, random UV sample, edge and vertex. Complete per-face triangle locators plus inspection-mesh bounding-box lower bounds select source hints without holding a global copied VTK mesh. Those hints are never CAD geometry; actual support/trim checks and full-boundary fallback remain required.

The separate method is `actual_trimmed_native_boundary_uv_witness_low_memory`. On fallback it re-evaluates the returned owned vertex, in-range native edge parameter, or strict-IN face parameter; a face ON band resolves to actual trim edges. Source distances use the actual resulting point norm, while numerical CPP-to-support residuals are reported separately. This addresses the previously observed difference between cached kernel coordinates and owned geometric support coordinates. It is still a finite actual-point witness metric, not a global-nearest or Hausdorff certificate.

Diagnostics use `native_memory_mapped_arrays.v1`: hashed `.npy` files for the complete native mesh/UV/actual points/distances and a small source-sample `.npz`. All workspace files remain as evidence. Mappings close on return and exceptions, including when a diagnostic callback retains references. The previous in-memory API/method names remain default and their frozen reports are not rewritten. Requested density and formal solid/tolerance/ownership gates are not relaxed. The OCCT mesher itself still needs external resource guarding; disk storage does not impose a hard native-memory cap or guarantee every surface can be meshed. Installed-wheel independent verification and an actual large-case resource check are required before claiming this route resolves its memory blocker.

## Native mesh lifetime correction (0.3.0.dev11 candidate)

Dev10's new low-memory entry point encountered native access violations on two complete models. Its earlier passing test records remain historical evidence. Initial controls appeared to implicate post-copy `BRepTools.Clean`, but their periodic traceback intervals differed. Later same-model runs retaining the original cleanup and removing only periodic traceback dumps completed. The earlier claim of a uniquely isolated cleanup cause is withdrawn; the exact native failure mechanism remains unknown.

Dev11 keeps the private native triangulation alive until validation exits, preserving the original default APIs, actual geometry, complete sample density, formal gates and mapped-array cleanup. Keeping the OCCT cache increases live native memory. This conservative lifetime choice is retained, but is not described as a proven cure for native crashes. Its 89 producer tests, 171 independent tests and completed real-model process check remain valid within their recorded scope; another large model still failed with periodic traceback instrumentation. A completed process can also report a real geometry failure.

## Execution diagnostics correction (0.3.0.dev12)

Dev12 changes documentation, comments and version metadata only; its executable geometric validation logic is unchanged from dev11. During current Windows OCCT checks, use the public `progress` callback and an external resource monitor. Avoid `faulthandler.dump_traceback_later(..., repeat=True)` for these runs: it is an unresolved interference candidate, and paired real-model runs completed after removing that periodic watchdog without changing source, STEP, seed, precision or coverage. Passive `faulthandler.enable()` for actual faults remained enabled in those controls. This is operational guidance, not a claim that every periodic watchdog necessarily fails or that every access violation is cured.

Keep all failed-run evidence. Do not treat a first-chance "Windows fatal exception" line as the exact process termination point when later progress exists, and do not count an interrupted report as geometric PASS or FAIL. Retain external process/private-memory and system physical/commit reserves; disabling diagnostic stack sampling does not remove resource protection. The library does not silently cancel a caller's global diagnostics. Large-case completion and source accuracy still require their own full reports and independent acceptance.

## Exact native curve interval intersection (0.3.0.dev13)

OCCT's Bezier decomposition can extend one floating-point step beyond a valid native B-spline edge's trimmed interval. The sampler previously rejected that harmless overcoverage because its final endpoint was not exactly the edge endpoint. It now intersects decomposition spans with the unchanged native interval. For a clipped end span, the derivative bound comes from the complete original Bezier control hull and is scaled by the actual interval-length ratio; subsequent subdivisions keep that conservative full-span bound. Returned samples still evaluate the original edge at its exact native parameters.

This does not change an edge, move its endpoints, snap parameters, use a loose equality test or relax the requested geometric bound. Actual missing coverage, internal gaps/overlaps and insufficient subdivision budgets still fail, including one floating-point step of missing coverage. Reports list the clipped decomposition spans. Unclipped spans keep the previous adaptive control-hull algorithm; a clipped span can require more samples because its bound is conservative.

Source charts remain the caller's responsibility. A single outer radial envelope cannot recover reentrant walls or multiple material intervals; increasing its density does not restore omitted boundaries. Use original-position/triangle evidence to partition such regions and preserve their interfaces, then run complete source/native verification. Shading normals may help seed parameters but are not original CAD face IDs or exact surface derivatives. The wheel does not automatically infer topology, recover original design history or turn a local chart check into whole-component acceptance.

## Actual owned trim-candidate recovery (0.3.0.dev14)

On a valid edge, OCCT's distance solver can report a parameter beyond the exact trim while remaining inside its numerical shape-tolerance band. The strict `owned_extrema_point` evaluator still rejects this value. A finite owned-edge interval failure now has a typed `NativeEdgeTrimParameterError`; nonfinite parameters, foreign ownership and other errors are not converted into recoveries.

The low-memory verifier uses `owned_extrema_point_with_recovery` to perform a fresh projection constrained to the actual native interval and enumerate the actual curve's two endpoints and its owned topological vertices. It discards every out-of-interval projector result, evaluates retained points on the original native support, and selects a candidate using the query distance. It never accepts or clamps the illegal parameter, changes a curve, alters a trim or raises a topology/geometry tolerance. The original query-to-actual-point distance is then measured by the verifier. This is an actual-point upper-bound witness, not a certificate of the globally closest point.

All recovery records are streamed to `native_trim_parameter_recoveries.jsonl` in the fresh diagnostics directory and bound by hash in the report. They preserve the rejected parameter, exact interval, actual candidates, selected point and distance. Normal strict-evaluation paths, sampling density, source data, serial meshing, old default verification APIs and the 0.01 mm native-tolerance limit stay unchanged. A successful recovery does not make a component pass; the complete original source/native check and independent acceptance are still required. Prior failed runs and installed versions remain separate.

## 0.3.0.dev15：显式提取材料边界

`guided_mesh_cad.extract_material_boundary(shape, evidence=...)` 只处理已经有效、正体积、恰好一个 Solid 的原生结果。它可以移除全部面均为 `INTERNAL` 的开放内部壳，但必须先证明这些面的完整实际区域位于保留材料内或其边界上：在独立副本中把查询面改为 `FORWARD`，执行整个面减去材料实体，零新增 fuzzy tolerance；有任何残余面、边、顶点或计算未完成均拒绝。标签、角点、体积不变或非空说明文字均不能代替空间检查。内含判断使用输入现有原生精度，不声称实数集合的全域数学证明。

输出复用全部原有闭壳和有向面，逐 occurrence 核 `IsEqual`，保留位置、方向、多重性及封闭内腔；核输入 BRep 哈希不变、输出合法性和体积数值对照。多个实体、游离拓扑、混合方向/外部开放壳、无效模型或原生容差超过 0.01 mm 均拒绝，不选择最大实体、不补面、不调整容差。原来的读取器和 `require_solid` 仍然拒绝带开放内部壳的原件，不会自动调用此 API。

```python
from guided_mesh_cad import extract_material_boundary
from guided_mesh_cad.cad import require_solid

material, extraction_report = extract_material_boundary(
    boolean_result,
    evidence="记录该布尔结果的来源和显式材料边界提取目的",
)
require_solid(material)
# 继续对冻结的全部原始来源做双向验证；提取报告不是零件验收。
```

此 API 不证明内部面的产生意图、来源结构完整性、原始 CAD 特征树或几何精度。必须保留原件及提取报告，再验证全部原始来源，不能删减原始三角面来使输出通过。

## 0.3.0.dev17 candidate：仿射 pcurve 的原生曲面边

新增显式接口 `pullback_affine_pcurve` 和 `make_affine_pcurve_edge`，不改变既有构造、采样或验收接口。输入是原生 `Geom_BSplineSurface` 支撑曲面和一个 UV 直线段；输出保留参数对应的有理三维 BSpline 曲线，或附实际仿射 pcurve 的 CadQuery 边。它们不推断 face 的裁剪域，不自动拼环、吞掉短边、合并折点或恢复设计历史。

```python
from guided_mesh_cad import make_affine_pcurve_edge

edge, report, certificate_arrays = make_affine_pcurve_edge(
    native_bspline_surface, uv_start, uv_end,
    parameter_range=(-2.5, 8.0),  # 必须有限、递增；UV方向可任意
    max_deviation_mm=1e-8, tolerance=1e-7,
    reverse=False,              # 仅改 TopoDS 方向，不改参数对应
    start_vertex=None, end_vertex=None,  # 可复用已核坐标/容差的真实端点
)
```

算法在实际跨越的全部 tensor Bezier 区间上作齐次 Bernstein 合成，再按被跨节点的原连续性求系数。真正的 C0 节点保留 C0；一条折线的每个真实线段须分别调用，不能把多个角点平滑成单边。等参线不无故升成双向次数和。不连续跨越、非正权、支撑域外端点、超过 OCCT 原生次数上限、不能表示全部区间的浮点参数范围或超出请求预算均明确拒绝。

每条原生区间都用正分母下界与 Bernstein 差异分子凸包计算完整同参数误差上界，并另列坐标转换及大偏置参数区间舍入的预算。`certificate_arrays` 保存源合成和实际原生有理 Bernstein 系数。它是浮点数学界计算，不是向外舍入的区间证明，也不是采样点误差冒充全域界。调用方应冻结曲面、参数、报告、数组和输出边，并独立核最终 face/solid 的所有原生边界、BOP 输入、全源双向几何和实际装配。

可选共享端点对应递增参数的起点/终点，与 `reverse` 无关；坐标偏移、既有顶点容差和已证曲线预算必须落在所请求原生容差内。接口不移动端点。拓扑容差不能超过0.01mm，也不能用更大容差替代原始来源精度检查。单边形象端点重合或小于拓扑可分辨范围时拒绝，由调用方核真实拓扑，接口不自动删除。

安装包自测、独立检查和工程验收分别记录；安装成功和单条边通过不能作为整件通过。


## 0.3.0.dev18 candidate: indexed signed winding

`build_winding_index(vertices, faces)` retains every triangle in an unchanged source soup.
Its `query(points, chunk_size=2000)` returns one interval and threshold state per point:
`1` means the interval is wholly in `abs(w)>0.5`, `-1` wholly in `abs(w)<0.5`, and `0`
means unresolved. These are winding states, not a physical-solid certificate for open,
self-intersecting or incomplete meshes. Mesh topology and actual contact distances
remain separate checks. Nonfinite query points stay unresolved.

The far-field gradient bound controls hierarchical approximation; floating-point
padding has finite validation, not a formal arbitrary-scale interval-arithmetic proof.
Ambiguous nodes are expanded to the original triangle `atan2` expression. No sampling,
distance cap, hole filling or dropped triangles are used. Original face order is retained.

NumPy is sufficient. Install the optional `acceleration` extra for Numba; results
report the loaded backend. The Python fallback has the same point coverage and winding
definition but is slower. Invalid face indices, empty meshes, malformed arrays and
nonpositive chunk/leaf sizes fail explicitly. Large/nonfinite arithmetic is not certified.

This development candidate requires installed-wheel producer tests and independent
acceptance before replacing the accepted dev17 wheel. Existing reconstruction methods,
tolerances and source/native-boundary checks are unchanged.

The candidate3 numeric boundary explicitly rejects lossy float64 coordinate
conversions, underflow-sensitive triangle scales and nonfinite hierarchy arithmetic.
Unsupported inputs raise `ValueError`; they do not return a definite OUT result.
This includes otherwise well-conditioned meshes below the supported floating-point
range. Supporting those scales would require a separately validated arithmetic path.

### 0.3.0.dev19 candidate — direct native BRep boundary validation

All three explicit boundary APIs (`validate_native_boundary`, `validate_native_boundary_uv_witness`, and `validate_native_boundary_low_memory`) accept `.brep`/`.brp` directly as well as `.step`/`.stp`. The existing positional and `step_path=` keyword signatures remain compatible. This extension does not alter the separate legacy `validate` CLI command.

BRep input uses the native BRep reader; there is no intermediate STEP export, repair or tolerance change. Reports identify `model_format`, `model_path`, `model_sha256`, `native_input_reader` and whether the input file stayed unchanged. Legacy `step_sha256` remains the same hash for STEP inputs and is null for BRep. A file change detected during reading or verification rejects the report. Original finite sampling, solid gates, full face/edge coverage, metrics and failure preservation are unchanged. A result for one format does not certify an exchanged copy; verify that copy separately.

### 0.3.0.dev20 candidate: explicit narrow-face meshing recovery

The three native-boundary APIs accept `recover_narrow_mesh_faces=True`. The default remains the prior strict native-mesher path. If an otherwise valid face has no usable UV triangulation and its width proxy `2*area/perimeter` indicates a narrower scale than the nominal mesh request, the opt-in path makes one native-only retry on a copied face at one quarter of that proxy. The retry is strictly finer; nominal modeling tolerance, angular precision, complete source coverage and all original edge/vertex witnesses stay unchanged. Successful UV mesh nodes and every triangle are retained, and the report records each retry's face, precision and counts.

Only auxiliary triangulation is transferred. A geometry serialization excluding mesh must stay byte-identical before/after; no surface, pcurve, topology, tolerance, INTERNAL entity or source file is repaired or removed. A non-narrow, invalid, zero-area or still-unmeshed face is rejected. The width proxy selects an algorithmic retry, not a certified local feature width or a guarantee that every face can be meshed. This option does not certify manufacturing, whole-source completeness or continuous motion.


## 0.3.0.dev21 candidate: native trim gap-trigger semantics

The actual native-boundary classifiers use OCCT 7.9.3's documented gap-check
trigger (0.1), with the existing 1e-9 point-query tolerance and bbox strategy.
This trigger is not a geometric acceptance tolerance. The previous explicit
1e-7 trigger incorrectly enabled special high-vertex-tolerance gap handling
on ordinary topology and could classify genuine material as OUT.
IN, ON and OUT handling, actual owned-point evaluation, complete point counts,
modeling limits, topology tolerances and mesh recovery options are unchanged.
Historical reports remain historical; regenerate affected metric evidence.
This candidate requires independent acceptance before publication.

## 0.3.0.dev22 candidate: explicit positive-material union checks

`union_positive_solids(named_inputs, expected_solids=None, inside_witnesses=None)`
returns `(shape, report)`. Supply an ordered mapping of names to complete valid
closed native shapes; an input may contain multiple positive-volume Solids.
It fuses one named operand at a time on independent copies, checking native
completion/errors, BRepCheck_Analyzer validity/computed shell closure, and the complete previous-body-minus-
result and operand-minus-result after each step. Every original input is also
checked against the final result. Any residual topology or unresolved native
operation rejects with `PatchTopologyError.report`; near-zero volume is never
used to declare an empty difference. Disconnected components are retained.

The conservative native-warning policy is `any_native_warning_is_unresolved`.
Any warning in a Fuse or material-difference operation raises `PatchTopologyError`
with status `unresolved_native_warning` and `native_material_preservation_checked=False`;
the actual native warning flag and text remain in that operation's diagnostic row.
This includes orientation/classification/build warnings even when the kernel
returns an empty valid difference. No warning is waived, automatically retried,
or resolved by finite inside witnesses. An unresolved warning is an evidence
limitation and does not prove the underlying geometry is wrong. A checked result
requires every native operation to complete without errors or warnings.

`validate_positive_union(result, named_inputs, ...)` exposes the same full final
preservation checks for an existing result. It checks containment of positive
inputs; it does not exclude extra result material. `expected_solids`, when
provided, is enforced on the final complete result. Optional `inside_witnesses`
maps input names to XYZ points, each required to be strictly IN the original
named input and the result; ON, OUT and UNKNOWN do not pass. No automatic center
or bbox witness is invented. Reports retain input hashes and verify originals
are unchanged, including rejection paths.

No clean, repair, largest-solid selection, fuzzy increase or native topology
tolerance adjustment occurs. Existing 0.01 mm native tolerance limits apply.
Native subtraction and point classifications are runtime evidence at existing
kernel precisions, not an exact-real mathematical proof, source geometry
acceptance, intended contact verification or a feature-completeness certificate.
The report's `valid` fields come from BRepCheck_Analyzer and do not mean complete
BOP geometry or self-interference checks passed. This API does not run
BRepAlgoAPI_Check SelfIntersect/SmallEdges or the full BOP geometry checks;
`native_self_interference_checked` and `full_BOP_geometry_acceptance` are
explicitly false. Callers must still run the applicable complete self-interference,
geometry and design checks. Basic native validity and positive-material retention
cannot replace those checks.
This development candidate needs installed-wheel tests and independent review;
the accepted dev21 wheel remains available separately.
