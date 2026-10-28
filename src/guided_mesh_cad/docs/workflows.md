# 可恢复工作流与短输出

## 无 CAD 的最小例子

创建 `input.txt`，再写同目录的 `plan.json`：

```json
{
  "schema": "guided-mesh-cad.workflow.v1",
  "tasks": [
    {"id": "source", "operation": "inspect-input", "inputs": {"file": "input.txt"}, "params": {}, "depends_on": [], "scope": "文件身份记录；不验收几何"},
    {"id": "receipt", "operation": "inspect-input", "inputs": {"file": {"task": "source", "artifact": "report"}}, "params": {}, "depends_on": [], "scope": "检查前一步报告文件"}
  ]
}
```

`run plan.json --out run-directory` 一次执行全部可运行任务，依赖失败则阻断其后继，独立任务可继续。产物引用自动建立依赖，不能引用不存在的产物或制造依赖环。任务 ID 是安全、唯一的标识，不接受路径穿越和 Windows 保留名。操作是白名单，不执行计划内的 shell 或任意 Python。

## 操作和输入

| operation | inputs | params |
|---|---|---|
| inspect-input | 一个或多个文件 | 可选 expected_sha256（输入名到摘要） |
| reconstruct | plan | 无；读取既有绑定网格的重建计划 |
| validate | mesh、step | tolerance_mm、samples、seed |
| diagnose-native | native | 无；保留既有严格原生门禁 |
| audit-domains | source、domains | 无；只核原三角面完整归属 |
| compare-native | source、target | tolerance_mm、source_matrix、target_matrix、target_instance、scope |
| assemble | plan | tolerance_mm |
| verify-assembly | plan、build_manifest | tolerance_mm；通常由运行器自动安排稳定 checkpoint_dir |

每个任务必须给出非空 scope。输入路径相对计划文件；产物引用用 `{ "task": "build", "artifact": "manifest" }`。旧重建计划的 mesh/construction_mesh、装配计划中的资源文件、验证所需 STEP 等间接输入也参与绑定。未知参数拒绝，不悄悄忽略。

原生交换 tolerance_mm 默认 0.01 mm，不能以提高该值绕过物理门禁。调用者的原始参考/简化范围原样保留。导出任务只签“导出完成”；整机验证必须重新完整导入 STEP，核全部实例，再检查各实例。

## 断点恢复

`resume RUN` 会重读当前计划，核源文件、参数/范围、依赖、包代码、运行时及已登记产物/传递证据的哈希。只有匹配且完整的成功记录才复用。源文件改变会使相关任务和后继失效；实现或运行时改变目前保守地使相关旧缓存失效，不宣称函数级精细影响分析。

中断任务重新建立独立 attempt，旧记录不覆盖；已完成的装配实例可按完整绑定恢复。已完成任务的 gzip 证据被改动或丢失也会失效；未损坏的实例可继续复用。稳定检查点只允许在该任务受控目录中，不接受外来文件夹冒充缓存。

确定性拒绝不会在每次 resume 时盲目重试。修正输入/实现后自动失效，或显式用 `resume RUN --retry-failed`。同一 RUN 有排他锁；同进程/不同进程同时恢复会拒绝冲突。不得把停止的整批任务写成成功，只继承其真实完成记录。

## 资源

计划 resources 和各任务 resources 可配置：

```json
{"memory_limit_mib":4096,"reserve_physical_mib":1536,"reserve_commit_mib":1536,"timeout_seconds":null,"poll_seconds":0.1}
```

MiB=1024² 字节；默认串行。内存上限与系统可用余量分别监测，上限不等于预分配。Windows 采用子进程 Job 内存上限和父进程监测；父进程退出时清理所属子进程，不操作其他任务或分页文件。超时默认关闭，显式配置后可中断。启动/运行资源不足保留记录，不自动忙循环。

## 按需诊断

```powershell
python -m guided_mesh_cad status RUN
python -m guided_mesh_cad diagnose RUN --task verify --view errors
python -m guided_mesh_cad diagnose RUN --task verify --view metrics
python -m guided_mesh_cad diagnose RUN --task verify --view evidence --offset 0 --limit 5
python -m guided_mesh_cad diagnose RUN --task verify --view log --offset 0 --limit 5
python -m guided_mesh_cad diagnose RUN --task verify --view report --pointer /records/0
```

默认每页 5 项，上限 100；单次返回有字节预算，过大的值给出文件指针或要求缩小查询范围。status 是已保存检查点的快照，resume 才重新验证缓存。读取状态/诊断不导入 numpy/CadQuery/OCP/VTK，不重算几何。

旧单步命令默认短摘要不超过 4096 UTF-8 字节；原生 stdout/stderr 收入独立日志，详细 JSON 仍保存。`--full` 是显式的大输出选择。工作流摘要保留计数、错误码、检查点及产物信息；不把完整点表反复传回 LLM。

轮子统计任务复用、实际点证书/距离回退和产物字节，不虚报模型 token 节省率。真实模型用量需调用端提供；独立检查、权限边界和验收标准不能为节省 token 删除。

## 来源特征域

`guided-mesh-cad.domains.v1` 绑定 source_sha256、units=mm 和 domains。每个域包含唯一 id、role（working/simplifiable/reference/unknown/excluded）、basis、原始 triangles 索引；简化/排除另外要求非空 authorization 字符串。每个原三角面必须有且只有一个域，不以删掉困难三角面改善统计。unknown 和遗漏保留未解决状态。

此接口仅检查来源完整归属，不判断设计意图、不验证声明授权的真实性、不验收几何偏差。功能和简化范围仍需调用者明确提供。
