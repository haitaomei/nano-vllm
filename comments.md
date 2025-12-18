# nano-vllm 内存管理与 KV 缓存机制深度解析

本文档旨在梳理 `BlockManager` 如何管理内存，以及 KV 缓存（KV Cache）在 nano-vllm 推理过程中的完整生命周期。

## 核心概念：PagedAttention

nano-vllm (以及 vLLM) 的核心是 PagedAttention 机制。我们可以将其类比为操作系统的虚拟内存和物理内存。

-   **物理内存 (Physical Memory)**：在模型加载时，系统会在 GPU 上预先分配一块巨大的、连续的内存空间，作为“物理 KV 缓存池”。这块内存被切分成固定大小的“物理块”（Physical Blocks）。
-   **虚拟内存 (Logical "Memory")**：每个推理请求（`Sequence`）都有一个逻辑上的 token 序列。这个序列也被逻辑地划分为“逻辑块”。
-   **页表 (Page Table)**：`BlockManager` 的主要职责就是维护一张“页表”（在代码中即 `sequence.block_table`），它负责将每个序列的“逻辑块”映射到“物理块”。

这种设计的最大优势是，序列的 KV 缓存可以在物理上不连续，从而彻底解决了内存碎片问题，极大地提高了 GPU 显存的利用率。

## 两阶段推理：Prefill 与 Decode

LLM 的生成过程分为两个截然不同的阶段：

| 特性 | Prefill (预填充) | Decode (解码) |
| :--- | :--- | :--- |
| **处理对象** | 整个输入 Prompt | 一次一个新生成的 Token |
| **计算方式** | 大规模并行计算 | 顺序、自回归计算 |
| **性能瓶颈** | **计算密集** (矩阵运算) | **访存密集** (读取 KV 缓存) |
| **发生频率** | 每个请求**一次** | 每个生成的新 Token **一次** |

---

## 完整工作流程示例

**场景设定:**

*   **模型配置**: `block_size = 4` (每个物理块能存 4 个 token 的 KV)。
*   **用户请求**: 输入 Prompt "Hello, my name" (4个 token)，模型需要生成下一个 token " is"。

### 第 1 步：处理 Prompt (Prefill 阶段)

1.  **调度决策 (`Scheduler`)**
    *   **动作**: `Scheduler` 收到新序列，计算出需要 `ceil(4/4) = 1` 个块。它调用 `block_manager.allocate(sequence)`。
    *   **结果**: `BlockManager` 从其空闲块列表中取出一个物理块 ID (例如 `42`)，并将其写入 `sequence.block_table`。
    *   **当前状态**: `sequence.block_table` 变为 `[42]`。

2.  **执行计算 (`ModelRunner`)**
    *   **动作**: `ModelRunner` 在 `prepare_prefill()` 方法中，为这 4 个 token 创建一个 `slot_mapping`，这个映射会精确地指向物理块 `42` 中的 4 个槽位。
    *   **结果**: 在 `Attention.forward()` 中，底层的 `store_kvcache` CUDA 核函数根据 `slot_mapping` 将这 4 个 token 的 K/V 值写入物理块 `42` 的相应位置。模型预测出下一个 token 是 " is"。

### 第 2 步：生成新 Token (Decode 阶段)

1.  **调度决策 (`Scheduler`)**
    *   **动作**: 序列长度变为 5，但其容量只有 4。`Scheduler` 发现容量不足，调用 `block_manager.append_slot(sequence)` 来申请一个新块。
    *   **结果**: `BlockManager` 分配一个新的物理块 ID (例如 `177`)。
    *   **当前状态**: `sequence.block_table` 变为 `[42, 177]`。

2.  **执行计算 (`ModelRunner` & `Attention`)**
    *   **读操作**: `ModelRunner` 在 `prepare_decode()` 中，将 `block_table` (`[42, 177]`) 传递给 `Attention` 层。`flash_attn_with_kvcache` CUDA 核函数利用这个“页表”，从物理块 `42` 中读取历史 K/V 数据。
    *   **写操作**: 同时，`ModelRunner` 为新 token " is" 创建 `slot_mapping`，指向物理块 `177` 的第 0 个槽位。`store_kvcache` 将新计算出的 K/V 值写入该位置。

---

## 关键组件职责与交互

*   **`BlockManager`**: **CPU 端的逻辑块管理器**。
    *   它不直接接触 GPU 内存，只管理一个包含物理块 ID (整数) 的空闲列表。
    *   它的主要职责是响应 `Scheduler` 的请求，填充 `sequence.block_table`。
    *   它是**层不可知 (Layer-Agnostic)** 的。

*   **`Scheduler`**: **决策者**。
    *   决定哪些序列可以运行、抢占或交换。
    *   在每个调度周期，它会检查序列的内存需求，并调用 `BlockManager` 来分配或释放物理块。

*   **`ModelRunner`**: **CPU 到 GPU 的桥梁**。
    *   在启动时，负责在 GPU 上分配真正的、巨大的、连续的物理 KV 缓存池。
    *   在执行时，负责将 `Scheduler` 和 `BlockManager` 的逻辑规划（如 `sequence.block_table`）翻译成 GPU CUDA 核函数能理解的执行计划（如 `slot_mapping` 和 `block_tables` 张量）。

---

## 多层注意力机制与全局 BlockManager 的协作

这是一个核心问题：全局的 `BlockManager` 如何与多层 `Attention` 一起工作？

答案是：**通过为每一层创建独立的 KV 缓存区，并使用通用的块索引来访问。**

### 内存结构：酒店比喻

*   **KV 缓存池 (`kv_cache`)**: 不是一个张量，而是一个**列表**：`[layer_0_cache, layer_1_cache, ..., layer_N-1_cache]`。
*   **酒店套房**: 想象一个 token 的 KV 缓存在物理上是一个“套房”，这个套房里有 N 个“卧室”，每个“卧室”专属于一个 Attention 层。
*   **`BlockManager`**: 酒店前台，只负责分配“楼层号”（物理块 ID，如 `42`），不关心楼层上的套房结构。
*   **`Attention` 层**: 第 `i` 层的 `Attention` 就是第 `i` 间卧室的主人。

### 工作流程

1.  **分配**: `BlockManager` 发出全局指令：“使用 42 号物理块”。
2.  **执行**: 在模型 `forward` 循环中：
    *   当处理第 `0` 层时，它接收到指令 `42`，然后访问 `layer_0_cache[42]`。
    *   当处理第 `1` 层时，它也接收到指令 `42`，然后访问 `layer_1_cache[42]`。
    *   ...
    *   以此类推。

**结论**: `BlockManager` 提供的物理块 ID (`42`) 是一个**通用索引**。`forward` 循环在迭代到每一层时，都使用这个**相同的索引**去访问**各自专属的** `layer_cache` 张量。这样就实现了 `BlockManager` 的层解耦和高效的内存访问。

**因此，`BlockManager` 管理的总块数 `num_blocks`，精确地等于每一层 `layer_cache` 张量的第一个维度的大小。**


# Note
* BlockManager 中对应的映射，在每个层都是一样的
