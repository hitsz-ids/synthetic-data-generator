# 概览

## 技术框架

以下是此项目涉及的技术列表。

|      技术       |   类别   |                                                                       目的                                                                       |
| :-------------: | :------: | :----------------------------------------------------------------------------------------------------------------------------------------------: |
|     PyTorch     | 深度学习 | 主流深度学习框架，提供动态计算图和自动微分。用于： - 构建和训练生成模型如VAE - GPU加速模型训练 - 实现自定义神经网络层和损失函数 - 模型保存和加载 |
|      NumPy      | 深度学习 |      基础科学计算库，版本限制用于保持稳定性。用于： - 高效多维数组运算 - 数据预处理和特征工程 - 数值计算和统计分析 - 与其他科学库的数据交换      |
|      SciPy      | 深度学习 |        基于NumPy的高级科学计算工具包。用于： - 高级统计分析和假设检验 - 概率分布计算和随机数生成 - 优化算法 - 稀疏矩阵运算和线性代数计算         |
|     Pandas      | 深度学习 |                  强大的数据分析和操作库。用于： - 结构化数据输入输出 - 数据清洗和预处理 - 复杂数据转换和聚合 - 时间序列数据处理                  |
|  scikit-learn   | 深度学习 |                  机器学习算法工具包。用于： - 数据预处理和特征缩放 - 模型评估和交叉验证 - 特征选择和降维 - 机器学习模型基准测试                  |
|      Faker      | 数据生成 |                    多语言虚拟数据生成库。用于： - 测试数据集生成 - 系统测试的模拟数据 - 示例数据生成 - 自定义数据生成规则支持                    |
|   Matplotlib    | 数据评估 |                            综合绘图库。用于： - 训练过程可视化 - 数据分布和统计绘图 - 模型评估可视化 - 报告和文档图形                            |
| table-evaluator | 数据评估 |                 专门的表格数据评估工具。用于： - 真实数据与合成数据的统计比较 - 数据生成质量评估 - 数据质量报告 - 分布比较可视化                 |
|     PyArrow     | 数据处理 |                 高性能数据处理库。用于： - 大规模数据的快速输入输出 - 内存高效的数据处理 - 与大数据工具的集成 - 列式数据格式处理                 |
|    Pydantic     | 数据处理 |                      数据验证和设置管理框架。用于： - 类型安全的配置加载 - API数据验证 - 模型参数验证 - 数据模式定义和验证                       |
|     loguru      | 日志记录 |                               现代日志工具。用于： - 训练过程日志记录 - 错误跟踪和调试 - 性能监控 - 结构化日志输出                               |
|   cloudpickle   | 数据处理 |             增强的Python对象序列化工具。用于： - 模型序列化和反序列化 - 复杂Python对象持久化 - 分布式计算中的数据传输 - 中间结果缓存             |
|     pluggy      | 插件系统 |                       Python插件框架。用于： - 实现可扩展架构 - 管理模型和处理器插件 - 支持自定义组件集成 - 实现模块化设计                       |
|     joblib      | 并行计算 |                              并行计算支持库。用于： - 数据处理并行化 - CPU密集型任务优化 - 结果缓存 - 并行模型训练                               |
|      Click      | CLI工具  |                                命令行界面框架。用于： - 构建CLI工具 - 参数解析和验证 - 子命令管理 - 用户交互界面                                 |

## 核心流程图

```mermaid
sequenceDiagram
    participant 用户
    participant 数据连接器
    participant 数据加载器
    participant 元数据
    participant 合成器
    participant 数据处理器
    participant 模型
    participant 评估器

    用户->>数据连接器: 创建连接器()
    数据连接器-->>数据加载器: 连接器
    用户->>数据加载器: 加载数据()
    数据加载器->>元数据: 从数据加载器获取()

    用户->>合成器: 拟合(元数据)
    合成器->>数据处理器: 转换(数据)
    数据处理器-->>合成器: 处理后数据
    合成器->>模型: 拟合(元数据, 处理后数据)
    模型-->>合成器: 训练后模型

    用户->>合成器: 采样(样本数量)
    合成器->>模型: 生成()
    模型-->>合成器: 合成数据
    合成器->>数据处理器: 反向转换(合成数据)
    数据处理器-->>合成器: 还原数据
    合成器-->>用户: 还原数据

    用户->>评估器: 评估(真实数据, 还原数据)
    评估器-->>用户: 评估结果
```

## 4+1 架构视图

### 逻辑视图

```mermaid
graph TB
    subgraph 基础设施层["基础设施层"]
        direction LR
        Arrow["Apache Arrow<br/>数据处理引擎"]
        PyTorch["PyTorch<br/>深度学习框架"]
        Sklearn["Sklearn<br/>机器学习框架"]
    end

    subgraph 核心["SDG核心库"]
        direction TB

        subgraph 数据引擎["数据引擎层"]
            数据加载器["数据加载器<br/>数据加载"]
            元数据["元数据<br/>元数据管理"]
            处理后数据["处理后数据<br/>统一数据格式"]
        end

        subgraph 处理层["处理层"]
            检查器["检查器<br/>数据检查"]
            处理器["处理器<br/>数据处理"]
            转换器["转换器<br/>特征转换"]
            格式化器["格式化器<br/>格式转换"]
        end

        合成器["合成器<br/>数据合成"]

        subgraph 模型层["模型层"]
            传统模型["传统模型"]
            大语言模型["大语言模型"]

            传统模型 --GAN网络--> CTGAN
            传统模型 --高斯Copula--> 高斯Copula

            大语言模型 --无数据生成--> 合成["数据合成"]
        end

        评估器["评估器<br/>数据评估"]

        %% 数据引擎内部关系
        数据加载器 --加载原始数据--> 处理后数据
        数据加载器 --提取--> 元数据
        元数据 --指导处理--> 处理后数据

        %% 处理层和数据引擎关系
        处理后数据 --输入--> 检查器
        检查器 --检查结果--> 处理器
        处理器 --处理后数据--> 转换器
        转换器 --转换后特征--> 格式化器

        %% 合成器关系
        元数据 --元数据配置--> 合成器
        格式化器 --标准化数据--> 合成器
        合成器 --控制--> 模型层
        模型层 --生成数据--> 合成器

        %% 评估器关系
        处理后数据 --原始数据--> 评估器
        合成器 --合成数据--> 评估器
    end

    subgraph 插件系统层["插件系统层"]
        direction LR
        连接器管理器["连接器管理器<br/>数据源管理"]
        处理器管理器["处理器管理器<br/>处理器管理"]
        检查器管理器["检查器管理器<br/>检查器管理"]
        模型管理器["模型管理器<br/>模型管理"]
        评估器管理器["评估器管理器<br/>评估器管理"]
    end

    %% 基础设施和核心依赖关系
    Arrow -.->|"提供列存储和计算"| 数据加载器
    PyTorch -.->|"提供深度学习训练"| CTGAN
    Sklearn -.->|"提供概率分布拟合"| 高斯Copula

    %% 插件系统和核心关系
    数据加载器 --注册--> 连接器管理器
    处理器 --注册--> 处理器管理器
    检查器 --注册--> 检查器管理器
    模型层 --注册--> 模型管理器
    评估器 --注册--> 评估器管理器
```

### 进程视图

```mermaid
flowchart TB
    subgraph SDG主流程[SDG主流程]
        direction TB
        命令行界面[命令行界面] --> 合成器

        subgraph 数据访问[数据访问层]
            数据连接器[数据连接器]
            数据加载器[数据加载器]
            元数据[元数据]
            处理后数据[处理后数据]
            连接器管理器[连接器管理器]
        end

        subgraph 数据处理[数据处理层]
            检查器[检查器]
            处理器[数据处理器]
            转换器[数据转换器]
            格式化器[数据格式化器]
            采样器[数据采样器]
            处理器管理器[处理器管理器]
        end

        subgraph 模型层[模型层]
            模型管理器[模型管理器]
            CTGAN模型[CTGAN模型]
            大语言模型[大语言模型]
        end

        合成器[合成器]
        评估器[评估器]
    end
```

### 开发视图

```mermaid
graph TB
    subgraph SDG包结构["SDG包结构"]
        direction TB

        subgraph 核心["sdgx"]
            数据模型["data_models<br/>(元数据和数据处理)"]
            模型["models<br/>(合成模型实现)"]
            数据连接器["data_connectors<br/>(数据源连接)"]
            命令行["cli<br/>(命令行接口)"]
            工具["utils<br/>(工具函数)"]
            类型["types<br/>(类型定义)"]
            异常["exceptions<br/>(异常定义)"]
        end

        subgraph 依赖["核心依赖"]
            PyTorch["torch>=2<br/>(深度学习)"]
            Arrow["pyarrow<br/>(数据处理)"]
            Sklearn["scikit-learn<br/>(机器学习)"]
            Pluggy["pluggy<br/>(插件系统)"]
            Pandas["pandas<br/>(数据分析)"]
            OpenAI["openai>=1.10.0<br/>(大语言模型接口)"]
        end
    end

    %% 核心包依赖关系
    数据模型 --使用--> 类型
    数据模型 --使用--> 异常
    模型 --使用--> 数据模型
    数据连接器 --使用--> 数据模型
    命令行 --使用--> 模型
    命令行 --使用--> 数据连接器

    %% 基础设施依赖关系
    模型 --深度学习--> PyTorch
    模型 --大语言模型--> OpenAI
    数据连接器 --数据处理--> Arrow
    数据连接器 --数据分析--> Pandas
    模型 --机器学习--> Sklearn
```

### 物理视图

```mermaid
graph TB
    subgraph 外部服务节点[外部服务节点]
        OpenAI[("OpenAI API服务<br/>>= 1.10.0")]
    end

    subgraph 分发注册节点[分发注册节点]
        PyPI[("PyPI仓库<br/>sdgx包")]
        Docker[("Docker仓库<br/>idsteam/sdgx")]
    end

    subgraph 计算节点[部署节点]
        subgraph 容器[Docker容器]
            SDGX1["SDGX服务"]
        end

        subgraph Python环境[Python运行环境]
            SDGX2["SDGX包"]
            PyTorch["PyTorch >= 2.0"]
            Arrow["Apache Arrow"]
        end
    end

    %% 部署关系
    PyPI -->|"pip安装"| Python环境
    Docker -->|"docker拉取"| 容器
    OpenAI -->|"API"| SDGX1
    OpenAI -->|"API"| SDGX2

    %% 运行时依赖
    SDGX2 --> PyTorch
    SDGX2 --> Arrow
```

### 场景视图

```mermaid
sequenceDiagram
    participant 用户
    participant 数据连接器
    participant 元数据
    participant 模型
    participant 评估器

    %% 场景1: 数据驱动合成
    rect rgb(200, 220, 240)
        Note over 用户,评估器: 场景1: 数据驱动合成
        用户->>数据连接器: 加载原始数据
        数据连接器->>元数据: 自动检测元数据
        元数据->>模型: 配置模型
        模型->>模型: 训练和生成
        模型->>评估器: 评估合成数据
        评估器->>用户: 返回评估结果
    end

    %% 场景2: 基于大语言模型的无数据合成
    rect rgb(220, 240, 200)
        Note over 用户,评估器: 场景2: 基于大语言模型的无数据合成
        用户->>元数据: 定义元数据
        元数据->>模型: 配置大语言模型
        模型->>模型: 生成合成数据
        模型->>用户: 返回生成结果
    end
```

# 开发指南

## 克隆项目

```sh
git clone https://github.com/hitsz-ids/synthetic-data-generator.git
```

## 创建 Python 环境

基于[miniconda](https://docs.anaconda.com/miniconda/) 创建环境。

```sh
conda create -n sdg python=3.11
conda activate sdg
```

## 代码风格和检查

我们使用 [black](https://github.com/psf/black) 作为代码格式化工具，最佳使用方式是安装 pre-commit 钩子，它会在每次提交前自动格式化代码。

在提交前安装 pre-commit

```bash
pip install pre-commit
pre-commit install
```

Pre-commit 会在每次提交前自动格式化代码，也可以手动对所有文件执行格式化

```bash
pre-commit run --all-files
```

注释风格遵循 [Google Python风格指南](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)。

## 本地安装

```sh
pip install -e '.[test,docs]'
```

## 单元测试

我们使用 pytest 编写单元测试，使用 pytest-cov 生成覆盖率报告

```bash
pytest -vv --cov-config=.coveragerc --cov=sdgx/ tests
```

在提交PR前请运行单元测试，**确保新功能已被单元测试覆盖**

注意，测试过程中会从 GitHub 下载测试数据，因此最好**开启全局代理**，以保证网络连通性

## 构建文档

安装文档依赖

```bash
pip install -e .[docs]
```

构建文档

```bash
cd docs && make html
```

使用 [start-docs-host.sh](dev-tools/start-docs-host.sh) 部署本地HTTP服务器来查看文档

```sh
cd ./dev-tools && ./start-docs-host.sh
```

访问 `http://localhost:8910` 查看文档。

## 开始了解功能

在理解本章概述中提到的所有内容后，我们建议从 SDG 功能开始进行下一步了解。你可以探索 `tests/` 包下的所有内容，使用大语言模型聊天工具（如 cursor ）来添加测试和被测试类，从而深入了解你想要理解的详细功能。

这里我们为大语言模型聊天工具提供一个系统角色提示，以帮助你提出好的问题。

```sh
请详细解释以下 Python 类的逻辑和实现。除了查看该类的代码外，你还需要查看被测试主体类的代码：

1. 描述该类的整体功能和目的，包括被测试的主体、其功能和基本使用逻辑。
2. 分析类中的每个方法，解释其功能和参数。
3. 解释类中使用的任何重要算法或设计模式。
```
