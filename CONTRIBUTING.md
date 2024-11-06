# Overview

## Technical

The following is a list of technologies involved in this project.

| Technology      | Category           | Purpose                                                                                                                                                                                                                                                                                        |
| --------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyTorch         | Deep Learning      | Mainstream deep learning framework providing dynamic computation graphs and automatic differentiation. Used for: - Building and training generative models like VAE - GPU-accelerated model training - Implementing custom neural network layers and loss functions - Model saving and loading |
| NumPy           | Deep Learning      | Fundamental scientific computing library with version constraint for stability. Used for: - Efficient multi-dimensional array operations - Data preprocessing and feature engineering - Numerical computation and statistical analysis - Data interchange with other scientific libraries      |
| SciPy           | Deep Learning      | Advanced scientific computing toolkit built on NumPy. Used for: - Advanced statistical analysis and hypothesis testing - Probability distribution calculation and random number generation - Optimization algorithms - Sparse matrix operations and linear algebra computations                |
| Pandas          | Deep Learning      | Powerful data analysis and manipulation library. Used for: - Structured data I/O - Data cleaning and preprocessing - Complex data transformation and aggregation - Time series data handling                                                                                                   |
| scikit-learn    | Deep Learning      | Machine learning algorithm toolkit. Used for: - Data preprocessing and feature scaling - Model evaluation and cross-validation - Feature selection and dimensionality reduction - ML model benchmarking                                                                                        |
| Faker           | Data Generation    | Multi-language fake data generation library. Used for: - Test dataset generation - System testing with mock data - Example data generation - Custom data generation rule support                                                                                                               |
| Matplotlib      | Data Evaluation    | Comprehensive plotting library. Used for: - Training process visualization - Data distribution and statistical plotting - Model evaluation visualization - Report and documentation graphics                                                                                                   |
| table-evaluator | Data Evaluation    | Specialized tabular data evaluation tool. Used for: - Statistical comparison between real and synthetic data - Data generation quality assessment - Data quality reporting - Distribution comparison visualization                                                                             |
| PyArrow         | Data Processing    | High-performance data processing library. Used for: - Fast I/O for large-scale data - Memory-efficient data processing - Integration with big data tools - Columnar data format handling                                                                                                       |
| Pydantic        | Data Processing    | Data validation and settings management framework. Used for: - Type-safe configuration loading - API data validation - Model parameter validation - Data schema definition and verification                                                                                                    |
| loguru          | Logging            | Modern logging utility. Used for: - Training process logging - Error tracking and debugging - Performance monitoring - Structured log output                                                                                                                                                   |
| cloudpickle     | Data Processing    | Enhanced Python object serialization tool. Used for: - Model serialization and deserialization - Complex Python object persistence - Data transfer in distributed computing - Intermediate result caching                                                                                      |
| pluggy          | Plugin System      | Python plugin framework. Used for: - Implementing extensible architecture - Managing model and processor plugins - Supporting custom component integration - Implementing modular design                                                                                                       |
| joblib          | Parallel Computing | Parallel computing support library. Used for: - Data processing parallelization - CPU-intensive task optimization - Result caching - Parallel model training                                                                                                                                   |
| Click           | CLI Tools          | Command-line interface framework. Used for: - Building CLI tools - Parameter parsing and validation - Subcommand management - User interaction interface                                                                                                                                       |

## Core Process Diagram

```mermaid
sequenceDiagram
    participant User
    participant DataConnector
    participant DataLoader
    participant Metadata
    participant Synthesizer
    participant DataProcessor
    participant Model
    participant Evaluator

    User->>DataConnector: create_connector()
    DataConnector-->>DataLoader: connector
    User->>DataLoader: load_data()
    DataLoader->>Metadata: from_dataloader()

    User->>Synthesizer: fit(metadata)
    Synthesizer->>DataProcessor: convert(data)
    DataProcessor-->>Synthesizer: processed_data
    Synthesizer->>Model: fit(metadata, processed_data)
    Model-->>Synthesizer: trained_model

    User->>Synthesizer: sample(n_samples)
    Synthesizer->>Model: generate()
    Model-->>Synthesizer: synthetic_data
    Synthesizer->>DataProcessor: reverse_convert(synthetic_data)
    DataProcessor-->>Synthesizer: restored_data
    Synthesizer-->>User: restored_data

    User->>Evaluator: evaluate(real_data, restored_data)
    Evaluator-->>User: evaluation_results
```

## 4+1 architectural view

### Logical view

```mermaid
graph TB
    subgraph Infrastructure["Infrastructure Layer"]
        direction LR
        Arrow["Apache Arrow<br/>Data Processing Engine"]
        PyTorch["PyTorch<br/>Deep Learning Framework"]
        Sklearn["Sklearn<br/>Machine Learning Framework"]
    end

    subgraph Core["SDG Core Library"]
        direction TB

        subgraph DataEngine["Data Engine Layer"]
            DataLoader["DataLoader<br/>Data Loader"]
            Metadata["Metadata<br/>Metadata Management"]
            ProcessedData["ProcessedData<br/>Unified Data Format"]
        end

        subgraph Processing["Processing Layer"]
            Inspector["Inspector<br/>Data Inspector"]
            Processor["Processor<br/>Data Processor"]
            Transformer["Transformer<br/>Feature Transformer"]
            Formatter["Formatter<br/>Format Converter"]
        end

        Synthesizer["Synthesizer<br/>Data Synthesizer"]

        subgraph Models["Model Layer"]
            Traditional["Traditional Models"]
            LLM["LLM Models"]

            Traditional --GAN Network--> CTGAN
            Traditional --Gaussian Copula--> GaussianCopula

            LLM --No Data Generation--> Synthesis["Data Synthesis"]
        end

        Evaluator["Evaluator<br/>Data Evaluator"]

        %% Data Engine Internal Relations
        DataLoader --Load Raw Data--> ProcessedData
        DataLoader --Extract--> Metadata
        Metadata --Guide Processing--> ProcessedData

        %% Processing Layer and Data Engine Relations
        ProcessedData --Input--> Inspector
        Inspector --Inspection Results--> Processor
        Processor --Processed Data--> Transformer
        Transformer --Transformed Features--> Formatter

        %% Synthesizer Relations
        Metadata --Metadata Config--> Synthesizer
        Formatter --Normalized Data--> Synthesizer
        Synthesizer --Control--> Models
        Models --Generated Data--> Synthesizer

        %% Evaluator Relations
        ProcessedData --Raw Data--> Evaluator
        Synthesizer --Synthetic Data--> Evaluator
    end

    subgraph PluginSystem["Plugin System Layer"]
        direction LR
        ConnectorManager["ConnectorManager<br/>Data Source Management"]
        ProcessorManager["ProcessorManager<br/>Processor Management"]
        InspectorManager["InspectorManager<br/>Inspector Management"]
        ModelManager["ModelManager<br/>Model Management"]
        EvaluatorManager["EvaluatorManager<br/>Evaluator Management"]
    end

    %% Infrastructure and Core Dependencies
    Arrow -.->|"Provide Column Storage and Computation"| DataLoader
    PyTorch -.->|"Provide Deep Learning Training"| CTGAN
    Sklearn -.->|"Provide Probability Distribution Fitting"| GaussianCopula

    %% Plugin System and Core Relations
    DataLoader --Register--> ConnectorManager
    Processor --Register--> ProcessorManager
    Inspector --Register--> InspectorManager
    Models --Register--> ModelManager
    Evaluator --Register--> EvaluatorManager
```

### Process view

```mermaid
flowchart TB
    subgraph SDGProcess[SDG Main Process]
        direction TB
        CLI[CLI Interface] --> Synthesizer

        subgraph DataAccess[Data Access Layer]
            DataConnector[Data Connector]
            DataLoader[Data Loader]
            Metadata[Metadata]
            ProcessedData[Processed Data]
            ConnectorManager[Connector Manager]
        end

        subgraph DataProcessing[Data Processing Layer]
            Inspector[Inspector]
            Processor[Data Processor]
            Transformer[Data Transformer]
            Formatter[Data Formatter]
            Sampler[Data Sampler]
            ProcessorManager[Processor Manager]
        end

        subgraph ModelLayer[Model Layer]
            ModelManager[Model Manager]
            CTGAN[CTGAN Model]
            LLM[LLM Model]
        end

        Synthesizer[Synthesizer]
        Evaluator[Evaluator]
    end
```

### Development view

```mermaid
graph TB
    subgraph SDGPackages["SDG Package Structure"]
        direction TB

        subgraph Core["sdgx"]
            DataModels["data_models<br/>(Metadata and Data Processing)"]
            Models["models<br/>(Synthesis Model Implementation)"]
            DataConnectors["data_connectors<br/>(Data Source Connection)"]
            CLI["cli<br/>(Command Line Interface)"]
            Utils["utils<br/>(Utility Functions)"]
            Types["types<br/>(Type Definitions)"]
            Exceptions["exceptions<br/>(Exception Definitions)"]
        end

        subgraph Dependencies["Core Dependencies"]
            PyTorch["torch>=2<br/>(Deep Learning)"]
            Arrow["pyarrow<br/>(Data Processing)"]
            Sklearn["scikit-learn<br/>(Machine Learning)"]
            Pluggy["pluggy<br/>(Plugin System)"]
            Pandas["pandas<br/>(Data Analysis)"]
            OpenAI["openai>=1.10.0<br/>(LLM Interface)"]
        end
    end

    %% Core Package Dependencies
    DataModels --Uses--> Types
    DataModels --Uses--> Exceptions
    Models --Uses--> DataModels
    DataConnectors --Uses--> DataModels
    CLI --Uses--> Models
    CLI --Uses--> DataConnectors

    %% Infrastructure Dependencies
    Models --Deep Learning--> PyTorch
    Models --LLM--> OpenAI
    DataConnectors --Data Processing--> Arrow
    DataConnectors --Data Analysis--> Pandas
    Models --Machine Learning--> Sklearn
```

### Physical view

```mermaid
graph TB
    subgraph ExternalServices[External Service Nodes]
        OpenAI[("OpenAI API Service<br/>>= 1.10.0")]
    end

    subgraph DistributionNodes[Distribution Registry Nodes]
        PyPI[("PyPI Registry<br/>sdgx package")]
        Docker[("Docker Registry<br/>idsteam/sdgx")]
    end

    subgraph ComputeNode[Deployment Node]
        subgraph Container[Docker Container]
            SDGX1["SDGX Service"]
        end

        subgraph PythonRuntime[Python Environment]
            SDGX2["SDGX Package"]
            PyTorch["PyTorch >= 2.0"]
            Arrow["Apache Arrow"]
        end
    end

    %% Deployment Relations
    PyPI -->|"pip install"| PythonRuntime
    Docker -->|"docker pull"| Container
    OpenAI -->|"API"| SDGX1
    OpenAI -->|"API"| SDGX2

    %% Runtime Dependencies
    SDGX2 --> PyTorch
    SDGX2 --> Arrow
```

### Scenarios

```mermaid
sequenceDiagram
    participant User
    participant DataConnector
    participant Metadata
    participant Model
    participant Evaluator

    %% Scenario 1: Data-Driven Synthesis
    rect rgb(200, 220, 240)
        Note over User,Evaluator: Scenario 1: Data-Driven Synthesis
        User->>DataConnector: Load Raw Data
        DataConnector->>Metadata: Auto-detect Metadata
        Metadata->>Model: Configure Model
        Model->>Model: Train and Generate
        Model->>Evaluator: Evaluate Synthetic Data
        Evaluator->>User: Return Evaluation Results
    end

    %% Scenario 2: LLM-Based No-Data Synthesis
    rect rgb(220, 240, 200)
        Note over User,Evaluator: Scenario 2: LLM-Based No-Data Synthesis
        User->>Metadata: Define Metadata
        Metadata->>Model: Configure LLM Model
        Model->>Model: Generate Synthetic Data
        Model->>User: Return Generated Results
    end
```

# Development Guide

## Clone project

```sh
git clone https://github.com/hitsz-ids/synthetic-data-generator.git
```

## Create python env

This is based on [miniconda](https://docs.anaconda.com/miniconda/).

```sh
conda create -n sdg python=3.11
conda activate sdg
```

## Code Style and Lint

We use [black](https://github.com/psf/black) as the code formatter, the best way to use it is to install the pre-commit hook, it will automatically format the code before each commit

Install pre-commit before commit

```bash
pip install pre-commit
pre-commit install
```

Pre-commit will automatically format the code before each commit, It can also be executed manually on all files

```bash
pre-commit run --all-files
```

Comment style follows [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

## Install Locally

```bash
pip install -e '.[test,docs]'
```

## Unit tests

We use pytest to write unit tests, and use pytest-cov to generate coverage reports

```bash
pytest -vv --cov-config=.coveragerc --cov=sdgx/ tests
```

Run unit-test before PR, **ensure that new features are covered by unit tests**

## Build Docs

Install docs dependencies

```bash
pip install -e .[docs]
```

Build docs

```bash
cd docs && make html
```

Use [start-docs-host.sh](dev-tools/start-docs-host.sh) to deploy a local http server to view the docs

```bash
cd ./dev-tools && ./start-docs-host.sh
```

Access `http://localhost:8910` for docs.

## Started with the features

After understanding all the content mentioned in the overview of this chapter, we recommend starting with the sdg functionality. You can explore everything under the tests/ package, using the LLM Chat Tool (such as cursor) to add test and tested classes, thereby gaining insight into the detailed functionalities you want to understand.

Here we provide a system-role prompt for the LLM Chat Tool to help you formulate good questions.

```sh
Please provide a detailed explanation of the logic and implementation of the following Python class. In addition to reviewing the code for this class, you also need to look at the code for the subject class being tested:
1. Describe the overall functionality and purpose of this class, including the subject being tested, its functions, and basic usage logic.
2. Analyze each method in the class, explaining its functionality and parameters.
3. Explain any important algorithms or design patterns used in the class.
```
