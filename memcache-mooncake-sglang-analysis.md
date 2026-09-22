# Ascend MemCache 与 Mooncake：底层能力和 SGLang 适配差异

分析日期：2026-09-21；最近修订：2026-09-22  
SGLang 基准：PR [#38827](https://github.com/sgl-project/sglang/pull/38827)，提交 `b223651`。  
分析方式：源码及公开文档检查，未进行硬件实测。

修订说明：已核实 review 所引用的源码为同一提交 `b2236514092acce1504aae094a93fa0829ac920f`。本次补充 DirectLinker、合法恢复边界集合和逐 rank 备份差异，并依据调用链明确 draft 对象键碰撞的触发条件。

2026-09-22 补充：重点整理 MemCache 已有但 SGLang 尚未调用或暴露的 API，并参考本次读取的 vLLM-Ascend `main` 实现补充 GVA 逐层传输流程（见 4.9）。vLLM 参考链接为浮动分支，不代表与 PR 同期的版本，也不代表旧版 MemCache 均支持这些接口；本文未验证具体依赖组合能否直接运行。

## 1. 结论与比较范围

Ascend MemCache 已具备分布式 KV 对象存储、批量读写、多 buffer 分层读写、设备内存读写、多副本等能力。SGLang PR #38827 接入了常规 HiCache L3 缓存查询、备份和恢复流程，但没有使用底层的全部能力，也未覆盖 Mooncake 在 SGLang 中的全部调用场景。

适配差异还涉及正确性：混合池可恢复边界的计算与传递、TP 分片的备份参与者，以及主模型与 draft 的对象键隔离，均需要优先于性能优化检查。

需要分别回答两个问题：

1. **底层能力差异**：MemCache 是否提供与 Mooncake 对应的存储和管理能力？
2. **框架适配差异**：MemCache 已提供的能力，是否被 SGLang 正确、充分地使用？

比较底层时，本文使用当前可访问的 MemCache 和 Mooncake 官方文档；它们可能比 PR 依赖的运行环境更新。部署前必须固定实际安装版本，不能把当前文档中的能力直接视为旧版本已经具备。

本文使用以下证据口径：

| 标记 | 含义 |
|---|---|
| 已确认 | 公开 API、源码或文档有直接依据 |
| 未发现对应能力 | 已检查的公开接口/配置未提供依据，不等于证明整个项目绝对没有 |
| 待验证 | 静态分析发现风险或接入可能性，需要运行测试或进一步追踪上游调用 |

## 2. SGLang 中的接入位置

PR 新增的 `NpuMemcacheStore` 直接继承 `HiCacheStorage`，与 `MooncakeStore` 并列；它不是 MooncakeStore 的子类，也不调用 Mooncake。

```text
SGLang 请求调度与前缀缓存
    ↓
HiCache 控制器
    ↓ HiCacheStorage 接口
    ├── MooncakeStore
    │     └── MooncakeDistributedStore
    └── NpuMemcacheStore
          └── memcache_hybrid.DistributedObjectStore
```

常规 L3 流程中，查询、写入和读取最终分别调用 `batch_is_exist`、`batch_put_from`、`batch_get_into`。HiCache 决定何时备份、预取和恢复；存储后端完成对象查询和数据读写。

除上图的常规 HiCache Store 路径外，同一提交还有以下独立接入点：

| 接入路径 | Mooncake | PR 中的 MemCache |
|---|---|---|
| 常规 HiCache Store | 已接入，经 host pool 备份和恢复 | 已接入 |
| Unified Cache external linker | 已有 `MooncakeDirectLinker`，直接访问设备 buffer，不经过 host cache tier | 未注册对应 linker |
| PD/Encoder Transfer Engine | 已有独立传输接入 | 此 PR 未接入 |

`registry.py` 在常规 HiCache 初始化分支之后，以 `elif` 选择 external linker，后端仅包括 Mooncake 和 Mori。DirectLinker 是已有的独立缓存接入路径，不能仅作为未来的设备直读写优化讨论。

配置 `--hicache-storage-backend npu_memcache` 只选择常规 L3 后端，不会自动接入 external linker，也不会替换 PD/Encoder 传输。

依据：[MemCache 适配器][sg-memcache]、[Mooncake 适配器][sg-mooncake]、[缓存注册入口][sg-registry]、[Mooncake DirectLinker][sg-direct-linker]、[共享 Transfer Engine][sg-engine]。

## 3. 维度一：MemCache 底层相对缺少或尚未确认的能力

### 3.1 服务端租户隔离与配额

**结论：Mooncake 有明确实现；MemCache 已检查的公开 API/配置中未发现等价能力。**

Mooncake 提供租户标识、租户配额准入和租户范围内的内存淘汰，需要服务端开启对应模式。MemCache 当前公开接口中未找到等价的服务端租户配额和隔离机制。

对 SGLang 的影响：多个业务共用缓存池时，仅给 key 添加前缀，可以降低命名冲突，却不能限制某个业务消耗多少容量，也不能提供按租户管理资源的语义。

注意：这是公开接口层面的结论。若 MemCache 存在其他版本或未公开接口，需要另行确认。

依据：[Mooncake 多租户部署][mc-tenant]、[MemCache Python API][mem-api]、[MemCache 配置文档][mem-config]。

### 3.2 多对象分组生命周期

**结论：Mooncake 有 `group_ids`；MemCache 公开接口未发现对应的多对象分组语义。**

一页模型 KV 可能拆成 K、V、索引或其他状态对象。Mooncake 允许为相关对象附加 group ID，使服务端按共享 TTL 和尽力而为的分组策略管理生命周期。它不是保证整组原子写入或可见性的事务接口。

没有类似分组时，相关对象可能独立淘汰，导致某些组件仍存在，但整页已经无法复用。HiCache 仍需检查完整性；分组主要影响缓存有效利用率和生命周期管理。

MemCache 的 `*_layers` 接口把多个 buffer 组织成一个对象，与“多个对象共享生命周期”是不同能力，不能混同。

依据：[Mooncake Store 的 Optional Object Groups][mc-store]、[MemCache API][mem-api]。

### 3.3 SSD 故障恢复限制

**结论：MemCache 已有 SSD 缓存，不能列为缺失；官方集成指南对所述 SSD 版本明确注明不支持故障恢复。**

vLLM-Ascend 官方指南要求 SSD 功能使用 `memcache_hybrid >= 1.2.0`，并在 SSD 初始化参数说明中注明恢复限制。该限制不能扩大为整个 MemCache 没有 HA，也不能推断后续版本始终没有恢复能力。

对 SGLang 的影响：写到 SSD 不等于重启或故障后必然能恢复有效 KV。需要单独测试存储进程重启、元数据恢复及缓存内容是否重新可见。

Mooncake 文档包含 SSD offload、提升和恢复相关机制，但其可用性同样取决于版本和配置，不应推导为无条件持久化保证。

依据：[MemCache SSD 部署指南][vllm-guide]、[Mooncake 存储部署指南][mc-deploy]。

### 3.4 不能归入“底层缺失”的项目

以下能力已有公开依据：

- 多副本及指定存放实例。
- 多 buffer 分层读写。
- 主机 DRAM 与设备 HBM 池化。
- 设备内存与存储池之间的数据读写。
- vLLM-Ascend 参考实现所调用的 GVA 分配、查询、拷贝及租约接口；写入完成接口有版本差异，见 4.9。
- LocalService 动态加入/移除，以及文档所述的元数据 HA 能力。
- SSD 缓存，受版本和部署条件限制。

PD/Encoder 传输也不能仅凭这个 PR 没接入就归为底层缺失。Mooncake 包含独立 Transfer Engine；Ascend 侧应进一步比较 MemCache 加 MemFabric 的能力组合和所需控制协议。

依据：[MemCache 项目说明][mem-readme]、[MemCache API][mem-api]。

## 4. 维度二：SGLang 适配缺口与正确性差异

本节分别标明“已有底层接口未使用”和“框架接入或正确性缺口”。后者不代表新增路径需要的所有底层语义均已得到确认。

### 4.1 多 buffer 分层读写

**结论：明确的适配缺口。**

MemCache 提供 `batch_put_from_layers` 和 `batch_get_into_layers`：每个 key 对应多个地址和长度。PR 的适配器仅调用普通的 `batch_put_from` 和 `batch_get_into`。

同版本 Mooncake 适配器包含多 buffer 元信息组织，并调用 `batch_put_from_multi_buffers` / `batch_get_into_multi_buffers`。因此可以参考其接入方式，扩展 MemCache 对离散数据布局的支持。

需要补充的工作不只是换函数名，还包括：

- 逻辑页到多 buffer 的地址映射。
- key、地址列表和长度列表的一致性检查。
- 页级结果与组件级结果之间的转换。
- 不同布局下的数据完整性和输出一致性验证。

依据：[MemCache API][mem-api]、[两种适配器源码][sg-memcache]、[Mooncake 多 buffer 处理][sg-mooncake]。

### 4.2 设备内存直接访问存储池

**结论：底层已有设备直读写；PR 的常规 HiCache Store 路径未使用。比较 Mooncake 时必须区分常规 Store 与 DirectLinker。**

MemCache 读写接口提供方向参数：`H2G/G2H` 对应主机内存与全局池，`L2G/G2L` 对应设备本地内存与全局池。这里的 L 不是 HiCache 的 L2。

PR 向存储接口传递 host buffer，未显式选择设备方向。典型路径仍是：

```text
NPU HBM ↔ 本地 CPU buffer ↔ 分布式存储池
```

底层可提供设备直读写，但要用于 HiCache，需要解决设备目标页分配、内存布局、计算同步和传输完成后的调度衔接，不能只替换一个参数。

同一提交的 Mooncake HiCache 常规路径也使用 host pool；但 Mooncake 已另外提供不经过 host cache tier 的 DirectLinker（见 4.6）。因此，MemCache 的差距既包括设备读写能力未被常规 Store 使用，也包括缺少对应 external linker 接入，不能只归为未来优化空间。

vLLM-Ascend 的 MemCache 后端已有设备方向及分层接口调用，可以参考，但不能直接证明移植到 SGLang 后收益一定为正。

依据：[MemCache API][mem-api]、[vLLM-Ascend MemCache 后端][vllm-backend]。

### 4.3 副本数量和对象放置策略

**结论：底层已有；PR 未显式传入策略。**

MemCache 的 `ReplicateConfig` 提供 `replicaNum` 和 `preferredLocalServiceIDs`。PR 写入只传 key、地址、大小，没有构造复制配置，因此没有从该适配层显式控制副本数和对象位置。

这可以通过增加 SGLang 存储配置解析并向底层透传来补充，随后验证副本形成、节点故障和容量开销。

需要注意：这是 MemCache 能力使用不足；不能仅因为 Mooncake 有 `ReplicateConfig` 类，就认定其 SGLang 适配器已暴露所有副本和放置策略。前述 Mooncake 分组能力与副本配置也不是同一功能。

本次查看的 vLLM-Ascend MemCache 写入封装也没有传入 `ReplicateConfig`，因此该项不能表述为“vLLM 已适配、SGLang 未适配”。

依据：[MemCache ReplicateConfig][mem-api]、[PR 写入实现][sg-memcache]。

### 4.4 模型命名空间自动隔离

**结论：相对 Mooncake 的明确适配差距，主要应在 SGLang 解决。**

Mooncake 适配器使用模型名以及可选 `extra_backend_tag` 构造前缀。MemCache 的 `_tag_keys()` 仅处理可选 tag。

多模型共用池时，这少了一层自动防冲突保护。实际是否冲突还取决于上游 key 的生成方式；不能只看字符串后缀就宣布已经发生错误。

建议补齐统一的模型/缓存格式标识。实验阶段至少显式配置 tag：需要共享的兼容实例使用同一 tag，不兼容的模型或缓存格式使用不同 tag。模型名本身也不是模型权重版本或完整 KV 格式的充分标识。

依据：[Mooncake key 前缀构造][sg-mooncake]、[MemCache `_tag_keys`][sg-memcache]。

### 4.5 Draft 和混合模型适配

**结论：有明确的分支覆盖差异；特定派生路径下可静态确认对象键碰撞，运行时影响尚未实测。**

MemCache v2 组件 key 生成主要处理 INDEXER、MAMBA 和 DRAFT。Mooncake 包含更多 pool 类型，并处理某些没有实体 `kv_buffer` 的逻辑节点。

Draft 分支尤其值得核对：

```text
Mooncake：…_<rank>_draft_k / …_<rank>_draft_v
MemCache：…_<rank>_k       / …_<rank>_v
```

同一提交中的调用链已经能够确认相同页 key 的来源：

1. `build_full_draft_pools()` 为 draft 设置 `indices_from_pool=PoolName.KV`。
2. `_resolve_sidecar_kv_derived_pool_transfers()` 在派生池未提供 keys 时，将其 keys 设为主 KV 的 `operation.hash_value`。
3. MemCache 主 MHA 页和 MHA draft 页使用相同的 `mha_suffix` 及 `_k`、`_v` 后缀；同一个 backend tag 不会把这两类对象区分开。

因此，在采用该派生路径、未提供独立 keys 且后缀一致的主 MHA＋MHA draft 组合中，可静态确认生成相同对象键的条件。实际触发的模型组合、读写结果和输出影响仍需验证，不能描述为已经完成硬件复现。

MemCache 写入还会跳过已存在的对象键，因此影响不一定是覆盖，也可能表现为将另一份数据误判为已经备份成功。此外，draft 分支固定生成 K/V 两个组件，没有像 Mooncake 那样按 draft 自身的 MLA/MHA 类型区分。

这些是 SGLang 如何组织对象的问题，不需要底层理解模型语义。混合模型适配还必须检查恢复边界和逐 rank 备份策略（见 4.7、4.8），不能只增加 Store 的 pool 分支。

依据：[MemCache 组件 key 与写入判断][sg-memcache]、[Mooncake 混合 pool 处理][sg-mooncake]、[draft 池装配][sg-draft-assembly]、[派生池 keys 解析][sg-derived-keys]。

### 4.6 Unified Cache DirectLinker 接入

**结论：Mooncake 已有对应 SGLang 接入，MemCache 尚未接入；vLLM-Ascend 已展示 MemCache 基于 GVA 的替代逐层方案，但移植所需的语义匹配仍需验证。**

`MooncakeDirectLinker` 注册设备内存 buffer，直接恢复设备侧缓存，通过逐层完成计数与计算同步。它的加载接口链为：

```text
batch_get_session_start(keys)
    → 对每层生成目标地址、长度和对象 offset
    → batch_get_into_multi_buffer_ranges(keys, ptrs, sizes, offsets)
    → 更新该层的完成状态
batch_get_session_end(keys)
```

这不是普通的多 buffer 全对象读取。MemCache 的 `*_layers` 本身不等于读取 session 或逐层调度。不过，vLLM-Ascend 已调用 `batch_alloc`、`batch_get_key_info`、`batch_copy` 及租约接口，通过 GVA 加偏移完成逐层传输（见 4.9）。因此不能再将替代方案笼统归为“底层是否支持尚不明确”；已有 API 调用证据，待确认的是它与 SGLang DirectLinker 所需生命周期和同步语义的匹配。

应分成两项工作：

- **SGLang 接入缺口（已确认）**：缺少 MemCache external linker 的注册、设备 pool 适配、逐层加载与完成/失败同步。
- **底层接口匹配（待验证）**：以 GVA、偏移拷贝和租约为候选方案，验证读写可见性、地址有效期、取消和异常清理；不要求先实现与 Mooncake 同名的 session/range API，也不预设两者语义完全等价。

这里“不经过 host cache tier”描述的是 SGLang 缓存层级，不代表承诺传输库内部任何协议都不使用临时 host buffer。

依据：[registry.py 的 external linker 分支][sg-registry]、[Mooncake DirectLinker 的加载实现][sg-direct-load]、[vLLM MemCache 后端][vllm-backend]、[vLLM 逐层传输实现][vllm-transfer]。

### 4.7 混合池的合法恢复边界集合

**结论：MemCache 的最大边界归约不足以表达有空洞的合法恢复集合；修复需覆盖 Store、控制器和跨 rank 同步。**

普通 `ALL_PAGES` 连续前缀可以用最大长度描述；`TRAILING_PAGES` 类型只要求某个结束位置前的窗口完整，可恢复边界可能不连续。

例如两个池的合法结束位置分别为：

```text
池 A：{2, 5}
池 B：{2, 4}

各池最大值取最小：min(5, 4) = 4    # A 不能在 4 恢复
合法集合求交后取最大：max({2}) = 2 # 共同可恢复边界
```

Mooncake 的 `batch_exists_v2()` 枚举各池合法边界并求交，返回 `PoolTransferResult(final_pages, hit_count, restorable)`。MemCache 找到各池最大边界后计算 `min(final_pages, boundary)`，返回值没有完整的 `restorable_prefix_pages` 集合。因此差异不只是返回字段少了一个，而是有空洞时可能选出某个池不支持的恢复位置。

还必须区分两级交集：

1. **Store 内跨池求交**：同一 rank 上所有必要状态必须支持该恢复位置。
2. **控制器跨 rank 求交**：各 rank 也必须共同支持该位置，不能只取各自最大值的最小值。

该版本传统 Hybrid 控制器在查询后只取 `hit_result.kv_hit_pages`，没有完整消费合法集合；Unified Cache linker 则已有按掩码跨 rank 求交的处理。因此，不能声称 Mooncake 在所有路径上都已解决该问题。补齐 MemCache 返回集合后，还需贯穿使用该结果的控制器、同步和后续恢复流程。

建议测试多池/多 rank 的有空洞集合、交集为空、窗口边界及全连续前缀对照，确认只在共同合法位置恢复。此问题属于框架恢复语义，不能归因于 MemCache 对象读写能力不足。

依据：[Mooncake 查询][sg-mooncake-boundaries]、[MemCache 查询][sg-memcache-boundaries]、[Hybrid 查询结果消费][sg-hybrid-query]、[Unified linker 跨 rank 求交][sg-linker-intersection]。

### 4.8 逐 rank 的分片备份参与者

**结论：控制器存在 Mooncake 专属的 draft 分片备份分支，MemCache 不能仅沿用主 KV 的跳过策略。**

基础控制器对 MLA 主 KV 设置 `backup_skip = is_mla_model and tp_rank != 0`，因为主 KV 在 TP 间复制，只需一个 rank 备份。但这不代表该请求的所有辅助状态也都是复制的。

Hybrid 控制器的 `should_backup()` 对 Mamba 单独放行；对 MHA draft / draft-SWA，则要求 `storage_backend_type == "mooncake"` 才检查并放行。MemCache 不满足该后端条件。

因此，在主 MLA KV 复制、draft MHA 按 TP 分片且走该控制器路径时，MemCache 非零 TP rank 的相关 draft 备份会被跳过。即使 Store 能正确编码这些对象，也可能没有收到需要写入的分片。draft-SWA 同时还需补齐其 Store 分支，不能把它视为已完整支持。

混合池适配必须同时检查：**对象编码、buffer 注册、恢复策略和逐 rank 备份参与者**。建议按 pool 的复制/分片属性判断备份责任，避免仅根据主模型类型或存储后端名称决定。

验证时应使用 TP 大于 1 的主 MLA＋MHA draft 组合，分别检查各 rank 的对象是否写入，并在本地缓存不命中时验证完整恢复。

依据：[基础控制器 backup_skip][sg-backup-skip]、[Hybrid should_backup][sg-should-backup]。

### 4.9 重点汇总：MemCache 已有、SGLang 尚未调用或暴露的 API

**核心结论：主要缺口是 SGLang 没有接入已有 Python API，而不是这些能力只有底层实现、尚无 Python API。** 其中最值得参考 vLLM-Ascend 的是多 buffer、设备直传和完整的 GVA 逐层传输流程。

#### 4.9.1 当前调用范围与缺口清单

SGLang PR 的核心数据操作如下，配合 host buffer 注册，实现常规 HiCache 的 L2↔L3 存取：

```python
store.batch_put_from(keys, buffer_ptrs, buffer_sizes)
store.batch_get_into(keys, buffer_ptrs, buffer_sizes)
store.batch_is_exist(keys)
```

| 底层已有能力 | API / 参数 | vLLM-Ascend 参考实现 | SGLang PR 的缺口 |
|---|---|---|---|
| 一个对象对应多个 buffer | `batch_put_from_layers`、`batch_get_into_layers` | 普通读写路径已调用 | 仍用单 buffer 接口，未利用多 buffer 对象表达 |
| 卡上 KV 与池之间直接传输 | `COPY_L2G`、`COPY_G2L` 方向参数 | 显式选择设备方向 | 使用 host pool 地址，未接设备直传路径 |
| 提前分配对象空间并取得全局地址 | `batch_alloc` | 为逐层写入准备 GVA | 没有对应的分配及地址管理流程 |
| 查询对象地址和大小 | `batch_get_key_info` | 加载前取得 GVA 等信息 | 主要查询是否存在，未接地址查询 |
| 按地址及偏移分段传输 | `batch_copy` | 计算各层地址后提交拷贝 | 没有基于 GVA 的逐层传输路径 |
| 读取期间的租约管理 | `batch_add_lease`、`batch_remove_lease` | 读取前获取、结束后释放 | 没有对应租约生命周期 |
| 分段写入后的完成通知 | `batch_write_finish` | 汇总写入结果后通知完成，兼容旧版无此接口 | 没有分段写入与完成通知流程 |
| 副本与对象放置策略 | `ReplicateConfig` | 本次查看的写入封装也未传入 | 未暴露并向写入接口传入策略 |

这里的“未接入”限定于该 PR 的 MemCache 路径，不表示 SGLang 整个框架没有逐层恢复能力。`L2G` 的 L 指本地卡上内存，不是 HiCache 的 L2 host cache。

多 buffer 与逐层传输也必须区分：前者允许一次调用把多个内存片段作为一个对象读写；后者根据模型执行进度分层提交传输，并管理同步及完成状态。**只把 `batch_put_from` 换成 `batch_put_from_layers`，不等于完成逐层适配。**

依据：[SGLang MemCache 适配器][sg-memcache]、[MemCache API][mem-api]、[vLLM MemCache 接口封装][vllm-backend]、[vLLM worker][vllm-worker]、[vLLM 传输线程][vllm-transfer]。

#### 4.9.2 最值得参考的 GVA 调用流程

GVA 是全局虚拟地址。vLLM-Ascend 的 MemCache 逐层路径可概括为以下流程，非可直接运行的完整代码：

```text
写入：
batch_alloc
  → 获得 GVA
  → 各层 batch_copy，使用对应偏移
  → batch_write_finish（支持该接口的版本）

读取：
batch_get_key_info
  → batch_add_lease
  → 各层 batch_copy
  → batch_remove_lease
```

`pool_worker.py` 负责分配及加载准备。其注释说明：读取侧不仅需要 GVA，还需要通过租约调用建立进程内地址追踪状态。因此不能只取得 GVA 就省略租约步骤。`kv_transfer.py` 按层计算地址和偏移、执行拷贝，并在最后一层读取完成后释放租约。

`batch_write_finish` 有版本差异：vLLM 后端动态检查该方法是否存在，注释说明旧版 MemCache 在 `batch_copy` 中发布写入。不能把完成通知接口列为所有版本都具备，也不能把兼容分支理解为所有版本具有完全相同的可见性保证。

对 SGLang 而言，这是一套可以研究用于 DirectLinker 的已有底层机制，而不是等待 MemCache 提供同名 Mooncake API。接入仍需实现设备 buffer 映射、逐层调度、结果检查、取消与异常清理，并验证地址和对象生命周期；已有参考代码不等于可直接替换或已经证明性能收益。

依据：[vLLM 分配及读取准备][vllm-worker]、[vLLM 拷贝与释放流程][vllm-transfer]、[写入完成接口的兼容处理][vllm-backend]。

#### 4.9.3 与其他差异的边界及建议顺序

- **已有 API 未使用**：上表列出的多 buffer、设备方向、GVA 流程和副本策略。
- **框架正确性问题**：draft key 冲突、合法恢复边界交集、逐 rank 备份责任。不能归为底层缺 API。
- **配置或验证不足**：SSD、独立服务及版本兼容。不能因为缺少专用 API 调用就判定不可用。

建议先修正确性，再补多 buffer 和副本配置，最后将设备直传与完整逐层生命周期作为独立适配工程。若目标是补齐 Mooncake DirectLinker 场景，应优先评估整套 GVA 流程，而不只是替换普通 Store 的读写函数。

## 5. 已有配置通道，不宜误判为未适配

PR 会将识别到的配置字段动态写入 `LocalConfig`，MetaService 启动脚本也采用字段映射。因此“README 没列出”或“没有专门的 if 分支”不足以证明功能不可用。

| 能力 | 更准确的判断 |
|---|---|
| DRAM/HBM 容量、传输协议、超时等 | 已有配置透传路径，取决于安装版本的字段 |
| 元数据 HA | 可由服务端部署配置提供，不要求每次读写有 HA 参数 |
| SSD 缓存 | 底层已支持；此 PR 未给出完整 SSD 接入验证，可能由存储服务透明提供 |
| 独立存储服务 | 底层有相关部署方式；不能用缺少 Mooncake 同名初始化分支证明无法使用 |

SSD 和独立服务的 SGLang 可用性仍需核对配置加载优先级、客户端初始化方式和版本兼容性，并做端到端测试。它们应列为“配置/验证覆盖不足或待确认”，不是已经证明的底层缺失。

依据：[PR 配置解析][sg-memcache]、[MetaService 启动脚本][sg-meta]、[MemCache 部署指南][vllm-guide]。

## 6. 建议优先级与验证方法

| 优先级 | 工作项 | 验证重点 |
|---|---|---|
| P0 | 模型/draft key 隔离与组件完整性 | 不兼容对象不串用，缺组件时正确回退 |
| P0 | 目标模型的混合 pool、布局、注册覆盖 | 注册成功，恢复数据与原始数据一致 |
| P0 | 合法恢复边界集合的端到端处理 | 跨池、跨 rank 求交；有空洞时不选非法边界 |
| P0 | 各 pool 的逐 rank 备份责任 | 主 MLA＋分片 MHA draft 下，非零 TP rank 的必要分片不被跳过 |
| P1 | 基础跨实例 L3 测试 | A 写入，B 本地未命中但从 L3 成功恢复 |
| P1 | 多 buffer API 接入 | 多地址对象读写正确，并测布局转换与传输开销 |
| P1 | 副本配置透传 | 副本策略生效、容量符合预期、故障行为可解释 |
| P1 | MemCache DirectLinker 接口评估与接入设计 | 参考 vLLM 的 GVA 分配、查询、拷贝、租约及完成通知，验证生命周期并设计设备 pool 和逐层同步 |
| P2 | SSD 与独立存储部署验证 | 配置实际生效，重启/故障行为符合版本承诺 |
| P2 | 设备直读写与 DirectLinker 性能验证 | 接入正确后，测逐层加载及减少中转的端到端收益 |
| 独立项目 | PD/Encoder 传输接入 | 传输接口、控制协议、状态同步及故障处理齐备 |

基础 L3 测试应使用相同模型与兼容配置：先由实例 A 预热固定长前缀，确认写入成功，再让从未处理过该前缀的实例 B 读取。观察实际存储读操作、恢复的页/字节数、有效缓存命中和首 token 延迟。反复请求 B 的同一前缀通常会转成本地 L1/L2 测试。

表中优先级为适配工作顺序；涉及特殊模型或路径的 P0 项在启用该组合前应完成。基础模型测试通过不代表混合池、draft 或 DirectLinker 路径已经正确。

不能仅用“启用 HiCache 后更快”证明 L3 生效，也不能仅凭存在 `batch_get_into` 调用证明恢复出的 KV 已被计算使用。

## 7. 仍需确认的事项

- 实际使用的 MemCache/MemFabric 版本与公开 master 文档的差异。
- MemCache 是否另有服务端租户配额及对象分组接口。
- 已静态确认的 draft key 碰撞条件在实际模型组合中的触发范围，以及跳过写入、读取失败或错误数据复用等运行时后果。
- vLLM 参考实现所需的 MemCache API/版本组合，尤其 `batch_write_finish` 的版本差异；GVA 与租约方案在 SGLang DirectLinker 中的同步、可见性及异常清理是否正确。
- 合法恢复集合从 Store 经控制器到跨 rank 同步是否完整传递，各 rank 的分片备份是否齐全。
- SSD 和独立服务在该 PR 初始化路径下的配置生效情况。
- 非基础模型的恢复正确性和性能，尤其 speculative decoding、混合 attention/state 模型。

本文不据此判断两种底层的绝对吞吐优劣。性能需要固定硬件、模型、数据布局、缓存命中范围、并发和存储配置后测量。

## 参考资料

- [SGLang PR #38827](https://github.com/sgl-project/sglang/pull/38827)
- [PR 基准 NpuMemcacheStore][sg-memcache]
- [同提交 MooncakeStore][sg-mooncake]
- [同提交 Mooncake Transfer Engine][sg-engine]
- [同提交缓存注册入口][sg-registry]
- [Mooncake DirectLinker][sg-direct-linker]
- [Hybrid 控制器][sg-hybrid-query]
- [draft 池装配][sg-draft-assembly]
- [Ascend MemCache 项目说明][mem-readme]
- [Ascend MemCache Python API][mem-api]
- [Ascend MemCache 配置文档][mem-config]
- [vLLM-Ascend MemCache 后端][vllm-backend]
- [vLLM-Ascend KV Pool worker：GVA 分配与读取准备][vllm-worker]
- [vLLM-Ascend KV 传输：逐层拷贝、完成通知与租约释放][vllm-transfer]
- [vLLM-Ascend KV Pool 部署指南][vllm-guide]
- [Mooncake 多租户部署][mc-tenant]
- [Mooncake Store 设计][mc-store]
- [Mooncake 存储部署指南][mc-deploy]

[sg-memcache]: https://github.com/sgl-project/sglang/blob/b223651/python/sglang/srt/mem_cache/storage/npu_memcache/npu_memcache_store.py
[sg-mooncake]: https://github.com/sgl-project/sglang/blob/b223651/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py
[sg-engine]: https://github.com/sgl-project/sglang/blob/b223651/python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py
[sg-meta]: https://github.com/sgl-project/sglang/blob/b223651/python/sglang/srt/mem_cache/storage/npu_memcache/start_meta_service.py
[sg-registry]: https://github.com/sgl-project/sglang/blob/b223651/python/sglang/srt/mem_cache/registry.py#L190
[sg-direct-linker]: https://github.com/sgl-project/sglang/blob/b223651/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_direct_linker.py
[sg-direct-load]: https://github.com/sgl-project/sglang/blob/b223651/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_direct_linker.py#L289
[sg-draft-assembly]: https://github.com/sgl-project/sglang/blob/b223651/python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py#L1118
[sg-derived-keys]: https://github.com/sgl-project/sglang/blob/b223651/python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py#L756
[sg-mooncake-boundaries]: https://github.com/sgl-project/sglang/blob/b223651/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L831
[sg-memcache-boundaries]: https://github.com/sgl-project/sglang/blob/b223651/python/sglang/srt/mem_cache/storage/npu_memcache/npu_memcache_store.py#L470
[sg-hybrid-query]: https://github.com/sgl-project/sglang/blob/b223651/python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py#L587
[sg-linker-intersection]: https://github.com/sgl-project/sglang/blob/b223651/python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py#L225
[sg-backup-skip]: https://github.com/sgl-project/sglang/blob/b223651/python/sglang/srt/managers/cache_controller.py#L553
[sg-should-backup]: https://github.com/sgl-project/sglang/blob/b223651/python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py#L717
[mem-readme]: https://github.com/Ascend/memcache/blob/master/README.md
[mem-api]: https://github.com/Ascend/memcache/blob/master/doc/memcache_python_api.md
[mem-config]: https://github.com/Ascend/memcache/blob/master/doc/memcache_config.md
[vllm-backend]: https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py
[vllm-worker]: https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py
[vllm-transfer]: https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py
[vllm-guide]: https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/kv_pool.md
[mc-tenant]: https://kvcache-ai.github.io/Mooncake/deployment/multi-tenancy.html
[mc-store]: https://kvcache-ai.github.io/Mooncake/design/store/mooncake-store.html
[mc-deploy]: https://kvcache-ai.github.io/Mooncake/deployment/mooncake-store-deployment-guide.html
