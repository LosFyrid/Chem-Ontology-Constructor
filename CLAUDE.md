# CLAUDE.md

本文件为Claude Code (claude.ai/code)在此仓库中工作时提供指导。

## 项目概述

Chem-Ontology-Constructor是一个复杂的AI驱动系统，用于从文本数据构建化学本体。该系统使用DSPy框架构建结构化LLM程序，并与OWL本体集成进行语义知识表示。

## 关键开发命令

### 测试
```bash
# 运行单元测试（使用pytest框架）
pytest tests/unit_test/
pytest tests/unit_test/query/  # 专门测试查询团队模块

# 运行特定测试文件
pytest tests/unit_test/query/test_query_agents.py
pytest tests/unit_test/query/test_ontology_tools.py
```

### 环境设置
```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 设置PROJECT_ROOT环境变量（配置解析所需）
export PROJECT_ROOT=/path/to/Chem-Ontology-Constructor/
```

### 配置
- 配置通过`config/settings.yaml`管理，支持环境变量替换
- 必须在settings.yaml中配置Java路径用于OWL推理（owlready2依赖）
- 需要OpenAI API密钥 - 通过`OPENAI_API_KEY`环境变量设置

## 架构概述

### 核心框架
系统围绕三种主要架构模式构建：

1. **多智能体工作流系统** (`autology_constructor/idea/`)
   - 使用LangGraph编排多智能体工作流
   - 三个主要团队：查询团队、梦想家团队、批评家团队
   - 通过类型化字典和状态处理器进行状态管理

2. **DSPy本体提取** (`autology_constructor/modules.py`)
   - 针对不同本体组件的模块化提取器（实体、元素、属性）
   - 带有奖励函数的精炼系统用于质量改进
   - 用于评估提取质量的评估框架

3. **OWL集成** (`config/settings.py`, 本体工具)
   - owlready2集成用于语义推理
   - 本体IRI的命名空间管理
   - 支持单领域和跨领域分析

### 关键组件

#### 查询团队 (`autology_constructor/idea/query_team/`)
- `query_agents.py`: 用于本体查询的LLM驱动智能体
- `ontology_tools.py`: OWL本体操作和分析工具
- `query_manager.py`: 协调查询执行和结果处理
- `query_workflow.py`: 查询处理的LangGraph工作流

#### 梦想家团队 (`autology_constructor/idea/dreamer_team/`)
- 多个发现器类用于识别研究空白和机会
- 证据、知识、方法论和元科学发现器
- 领域分析和跨领域比较能力

#### 状态管理 (`autology_constructor/idea/state_manager.py`)
- 具有类型安全的通用状态处理器框架
- 针对每个团队的专门结果处理器（查询、梦想家、批评家）
- 创建适当处理器的工厂模式

### 数据结构
- `base_data_structures.py`: 核心本体数据结构（实体、属性、元素）
- `signatures.py`: 结构化LLM交互的DSPy签名
- 使用TypedDict和数据类的类型安全方法

## RIPER-5协议合规性

此项目使用RIPER-5严格操作协议（定义在`.cursor/rules/riper-5-en.mdc`中）。在处理此代码库时：

1. **始终以模式声明开始响应**: `[MODE: MODE_NAME]`
2. **遵循五模式进程**: RESEARCH → INNOVATE → PLAN → EXECUTE → REVIEW
3. **禁止未授权修改** - 仅实现明确计划和批准的内容
4. **状态转换需要明确的用户许可**

## 文件组织模式

- 配置: `config/` (settings.yaml, settings.py)
- 主库: `autology_constructor/` (核心功能)
- 测试: `tests/` (单元测试、集成测试、测试数据)
- 数据: `data/` (本体、数据集、示例)
- 源工具: `src/` (辅助智能体和工具)

## 重要依赖

- **DSPy**: 用于结构化LLM编程和优化
- **owlready2**: OWL本体操作和推理
- **LangGraph**: 多智能体工作流编排
- **PyYAML**: 带变量替换的配置管理
- **pytest**: 测试框架

## 本体处理

- 本体文件以.owl格式存储在`data/ontology/`中
- 单个.owl文件触发单领域分析
- 多个.owl文件触发跨领域分析
- 推理操作需要Java路径配置
- 基础IRI模式: `http://www.test.org/chem_ontologies/`

## 测试策略

项目使用pytest，为每个组件提供特定的测试模块：
- 查询团队测试在`tests/unit_test/query/`中
- LLM交互的基于mock的测试
- 配置驱动的测试数据管理
- 不同分析模式的独立测试环境