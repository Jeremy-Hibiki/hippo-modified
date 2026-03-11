# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是 HippoRAG 的修改版本，一个基于知识图谱的检索增强生成 (RAG) 系统。该项目通过 Open Information Extraction (OpenIE) 从文档中提取实体和关系，构建知识图谱，并使用图遍历算法（如 Personalized PageRank）进行检索。

## 核心架构

### 主要组件

1. **客户端 (`hippo_modified/client/`)**
   - `HippoRAG`: 同步版本的主要客户端类
   - `AsyncHippoRAG`: 异步版本的主要客户端类
   - `BaseHippoRAG`: 共享的基础类，包含初始化逻辑和通用方法
   - `HippoRAGProtocol`: 定义客户端接口的协议类

2. **信息提取 (`hippo_modified/information_extraction/`)**
   - `OpenIE`: 使用 LLM 从文本中提取命名实体和关系三元组
   - 支持 NER (Named Entity Recognition) 和三元组提取
   - 使用 `json-repair` 库修复损坏的 JSON 响应

3. **嵌入模型 (`hippo_modified/embedding_model/`)**
   - `BaseEmbeddingModel`: 嵌入模型的抽象基类
   - 支持 OpenAI 兼容的嵌入 API

4. **嵌入存储 (`hippo_modified/embedding_store/`)**
   - `BaseEmbeddingStore`: 存储抽象基类，定义了同步和异步接口
   - `DataFrameEmbeddingStore`: 基于 DataFrame 的内存存储
   - `MilvusEmbeddingStore`: 基于 Milvus 的向量数据库存储（可选依赖）
   - 支持三种命名空间：`chunk`、`entity`、`fact`

5. **LLM 接口 (`hippo_modified/llm/`)**
   - `CacheOpenAI`: OpenAI API 的包装器，支持响应缓存
   - 支持 Azure OpenAI 和自定义端点
   - 使用 SQLite (同步) 和 aiosqlite (异步) 进行缓存

6. **提示模板 (`hippo_modified/prompts/`)**
   - `PromptTemplateManager`: 管理提示模板，支持角色映射
   - 模板存储在 `prompts/templates/` 目录下的 Python 文件中
   - 每个 `.py` 文件应包含一个 `prompt_template` 变量

7. **工具 (`hippo_modified/utils/`)**
   - `config_utils.py`: `BaseConfig` 数据类，包含所有配置选项
   - `typing.py`: 类型定义和别名
   - `misc_utils.py`: 通用实用函数
   - `logging_utils.py`: 日志配置

### 工作流程

1. **索引阶段**:
   - 文档被分块并使用 OpenIE 提取实体和关系
   - 提取结果保存到 `{save_dir}/openie_results.json`
   - 构建知识图谱（使用 igraph）
   - 为 chunks、entities 和 facts 创建嵌入并存储

2. **检索阶段**:
   - 从查询中提取命名实体
   - 在嵌入存储中搜索相关实体
   - 使用 Personalized PageRank 在图谱上遍历
   - 检索相关的 passages 和 facts
   - 可选的重排序（使用 DSPy）

## 开发命令

### 代码质量检查

```bash
# 运行 linter（自动修复问题）
ruff check

# 格式化代码
ruff format

# 类型检查
mypy hippo_modified

# 运行所有检查（使用 pre-commit）
pre-commit run --all-files

# 安装 pre-commit hooks
pre-commit install
```

### 包管理

该项目使用 `uv` 作为包管理器：

```bash
# 安装依赖
uv sync

# 添加开发依赖
uv sync --group dev

# 构建
uv build
```

### 可选依赖

```bash
# 安装 Milvus 支持
uv sync --extra milvus
```

## 配置系统

所有配置通过 `BaseConfig` 数据类管理。关键配置选项：

- **LLM 配置**: `llm_name`, `llm_base_url`, `llm_generate_params`
- **嵌入配置**: `embedding_model_name`, `embedding_base_url`, `embedding_use_instruction`
- **存储配置**: `embedding_store_type` ("dataframe" 或 "milvus"), `milvus_uri`
- **缓存**: `cache_llm_response`, `force_openie_from_scratch`, `force_index_from_scratch`
- **图遍历**: `passage_node_weight` (控制 PPR 中 passage 节点的权重)

配置可以在初始化时通过参数传递，或直接传递 `BaseConfig` 实例。

## 同步与异步

项目同时提供同步 (`HippoRAG`) 和异步 (`AsyncHippoRAG`) 实现：
- 异步版本使用 `asyncio.Lock` 防止并发准备问题
- 所有嵌入存储方法都有异步对应版本
- LLM 调用在异步版本中使用 `aiosqlite` 进行缓存

## 命名约定

- chunk IDs: 使用 `compute_mdhash_id(passage, "chunk-")` 生成
- 文档路径: `{save_dir}/openie_results.json`
- Milvus 集合名: `{milvus_collection_prefix}_{namespace}`

## 类型提示

项目使用严格的类型检查：
- 所有文件必须以 `from __future__ import annotations` 开始
- 使用 `TYPE_CHECKING` 导入类型提示
- Pydantic 模型用于数据验证
- 运行时评估的基类和装饰器已在 ruff 配置中正确设置

## 代码风格

- 行长度: 119 字符
- 目标 Python 版本: 3.10+
- 使用 ruff 进行 linting 和格式化
- mypy 用于静态类型检查
