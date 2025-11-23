````markdown
# MitoGPT - DNABERT-2 Fine Tuning for Human Mitochondrial Gene Tasks

This repository contains the complete data, training, and analysis pipeline for fine tuning the DNABERT-2 genomic foundation model on curated human mitochondrial sequence tasks. The primary supervised task is human mitochondrial gene type classification, where sequence windows are assigned to one of three classes: coding sequence (cds), ribosomal RNA (rRNA), or transfer RNA (tRNA).

The project is part of the broader MitoGPT effort to construct mitochondria-aware genomic language models for downstream annotation and variant reasoning.

---

## Repository contents

### Notebooks

- `MitoGPT_Data_Explorer.ipynb`  
  Interactive notebook for exploration of curated mitochondrial datasets. Typical operations include:
  - Loading Parquet or similar tables into pandas DataFrames  
  - Inspecting schema and column names  
  - Visualizing label distributions and class balance  
  - Plotting sequence length histograms and other quality control views  

- `MitoGPT_Data_Pipeline_Full.ipynb`  
  End-to-end data preparation pipeline. Main responsibilities:
  - Loading raw or intermediate mitochondrial sequence and annotation sources  
  - Filtering and normalizing annotations  
  - Constructing task-specific tables, for example `human_gene_type`  
  - Creating train and validation (and optionally test) splits  
  - Writing standardized Parquet datasets and metadata suitable for Hugging Face Datasets and the fine tuning notebook  

- `MitoGPT_Fine_Tuning_Pipeline_Full.ipynb`  
  Full fine tuning pipeline for DNABERT-2 on mitochondrial tasks. This notebook:
  - Loads DNABERT-2 configuration and tokenizer  
  - Converts curated tables into Hugging Face `Dataset` objects  
  - Tokenizes DNA sequence windows  
  - Configures `TrainingArguments` and the `Trainer` API  
  - Trains a three-class sequence classification head  
  - Evaluates on a held-out validation set  
  - Saves metrics, checkpoints, and training state to disk  

### Configuration and tokenizer

- `config.json`  
  Model configuration file for the DNABERT-2 based sequence classifier. Contains:
  - Architecture type (Bert-like)  
  - Number of layers, hidden size, attention heads  
  - Dropout settings  
  - Vocabulary size  
  - Number of labels and label mapping for the classification task  

- `configuration_bert.py`  
  Python module that defines a configuration class compatible with `config.json`. This file implements any DNABERT-2 or Mosaic-style extensions needed by the `transformers` library, such as custom attention options.

- `tokenizer.json`  
  Serialized DNABERT-2 tokenizer. Implements a BPE-style vocabulary over genomic sequences with a typical vocabulary size of 4096 subword tokens that capture variable length nucleotide motifs.

- `tokenizer_config.json`  
  Configuration for the tokenizer. Specifies:
  - Tokenizer class name  
  - Maximum sequence length  
  - Padding and truncation behavior  
  - Special token identifiers and their behaviors  

- `special_tokens_map.json`  
  Explicit mapping of special tokens:
  - `[PAD]` padding token  
  - `[CLS]` classification token  
  - `[SEP]` separator token  
  - `[MASK]` mask token  
  - `[UNK]` unknown token  

These files allow reconstruction of the exact tokenizer and model configuration used during fine tuning.

### Training outputs and checkpoints

- `model.safetensors`  
  Trained DNABERT-2 sequence classification weights stored in the `safetensors` format. This file is the central artifact for inference or further fine tuning.

- `optimizer.pt`  
  Serialized optimizer state, typically for AdamW. Enables exact resumption of training from the last checkpoint with correct momentum and adaptive statistics.

- `scheduler.pt`  
  Serialized learning rate scheduler state. Preserves the current position on the learning rate schedule so that a resumed run continues with the same schedule.

- `training_args.bin`  
  Serialized `TrainingArguments` object from the Hugging Face `transformers` library. Stores hyperparameters and runtime configuration such as batch size, number of epochs, logging steps, learning rate, warmup steps, and output directory.

- `trainer_state.json`  
  Human-readable log of the training progress. Includes:
  - Global step  
  - Metric history for training and evaluation loss  
  - Learning rate over time  
  - Evaluation checkpoints and epochs  

- `rng_state.pth`  
  Snapshot of the random number generator state for PyTorch (and optionally NumPy and Python). Supports bit-wise reproducibility when combined with the same environment and seeds.

- `baseline_metrics.json`  
  Final evaluation metrics of the main run on the validation set. Example content:

  ```json
  {
    "eval_loss": 0.0432724133,
    "eval_accuracy": 0.9912023460,
    "eval_f1_macro": 0.9796146313,
    "eval_f1_micro": 0.9912023460,
    "eval_runtime": 7.28,
    "eval_samples_per_second": 140.523,
    "eval_steps_per_second": 8.791,
    "epoch": 3.0
  }
````

These values summarize performance after three epochs of fine tuning.

Additional folders (not listed here) can hold prepared Parquet datasets, figures, model export directories, or documentation.

---

## Project objectives

The repository is designed to achieve several goals:

1. Construct a curated human mitochondrial dataset that supports gene type classification from sequence alone.
2. Demonstrate that a modern genomic foundation model, DNABERT-2-117M, can be adapted to this task with high performance.
3. Provide a reproducible training pipeline including exact configuration, tokenizer, checkpoints, and metric logging.
4. Serve as a starting point for more complex MitoGPT experiments, such as multi-task training, cross-species learning, and variant effect prediction.

---

## Task description

### Human mitochondrial gene type classification

* **Input**
  Short windows of DNA sequence from the human mitochondrial genome. The windows are typically extracted around annotated gene regions and formatted as plain nucleotide strings (A, C, G, T).

* **Output**
  A single gene type label for each window:

  * `cds`  coding sequence
  * `rRNA` ribosomal RNA gene region
  * `tRNA` transfer RNA gene region

This task evaluates whether DNABERT-2 can internalize sequence-level patterns that distinguish these three gene categories in the mitochondrial context.

---

## Results summary

The primary run captured in `baseline_metrics.json` and `trainer_state.json` yields:

* Validation loss around 0.043
* Validation accuracy around 0.991
* Macro-averaged F1 around 0.980
* Micro-averaged F1 around 0.991
* Evaluation runtime on the order of a few seconds with high throughput per second

These metrics indicate that the model accurately distinguishes cds, rRNA, and tRNA mitochondrial regions in the curated dataset. The small gap between macro and micro F1 shows that performance is balanced across classes rather than dominated by a single majority class.

Training logs in `trainer_state.json` show fast convergence within three epochs, with loss dropping from near 1 at the start of training to values close to zero. Occasional spikes in loss or gradient norm appear but do not affect final performance, which suggests a stable optimization process.

---

## Technologies used

Core software stack across the notebooks and scripts:

### Language and runtime

* Python 3
* Jupyter or JupyterLab for notebook execution
* Google Colab support through:

  * `google.colab.drive` for Drive mounting
  * `google.colab.files` for downloading artifacts

### Data and numerics

* `numpy` for dense numerical operations
* `pandas` for tabular data handling and exploratory analysis
* `pyarrow` and `pyarrow.parquet` for reading and writing Parquet datasets

### Machine learning

* `torch` (PyTorch) for tensor computation and automatic differentiation

* `transformers` for DNABERT-2 model handling and training:

  * `AutoConfig`
  * `AutoTokenizer`
  * `AutoModelForSequenceClassification`
  * `TrainingArguments`
  * `Trainer`
  * `DataCollatorWithPadding`
  * `set_seed`

* `datasets` for dataset abstraction and interaction with Parquet files:

  * `Dataset`
  * `DatasetDict`
  * `concatenate_datasets`

* `safetensors` for safe and efficient model weight serialization

### Evaluation and splitting

* `scikit-learn`:

  * `accuracy_score`, `f1_score`, and related metrics
  * `StratifiedKFold`, `KFold`, `GroupKFold`, `train_test_split` for split construction where needed

### Utilities and logging

* Standard library modules:

  * `os` and `pathlib` for filesystem operations
  * `json` for configuration and metrics serialization
  * `math` and `random` for numerical utilities and randomness
  * `gc` for manual garbage collection in long-running sessions
  * `re` for regular expression based text handling
  * `subprocess`, `sys`, `importlib` for environment control and dynamic imports

* `rich` for colored and structured console logging

* Visualization libraries such as:

  * `matplotlib` and optionally `seaborn` for plotting metrics, distributions, and training curves

This collection of tools supports the complete workflow from data ingestion to model training and analysis.

---

## Running the pipeline

### Environment setup

A minimal environment based on `pip` can be prepared as follows:

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch transformers datasets numpy pandas scikit-learn pyarrow safetensors rich jupyter matplotlib seaborn
```

For GPU acceleration, a CUDA-enabled PyTorch wheel is recommended.

### 1. Data exploration

1. Open `MitoGPT_Data_Explorer.ipynb` in Jupyter or Google Colab.
2. Configure data paths at the top of the notebook, pointing to the curated mitochondrial data.
3. Execute the cells in order to:

   * Load the dataset
   * Inspect columns and data types
   * Visualize label balance and sequence lengths
   * Validate that the dataset matches expectations for the downstream task

### 2. Data pipeline

1. Open `MitoGPT_Data_Pipeline_Full.ipynb`.
2. Configure locations of raw data and desired output directories.
3. Execute the notebook to:

   * Load raw mitochondrial sequences and annotations
   * Filter and normalize metadata
   * Create task-specific tables such as `human_gene_type`
   * Build train, validation, and optionally test splits
   * Save standardized Parquet datasets and any manifest files

The resulting Parquet files act as the canonical inputs for the fine tuning notebook.

### 3. Fine tuning DNABERT-2

1. Open `MitoGPT_Fine_Tuning_Pipeline_Full.ipynb`.

2. Set:

   * The base model identifier (for example `zhihan1996/DNABERT-2-117M`) or local config files
   * Paths to the train and validation Parquet datasets from the data pipeline
   * Hyperparameters such as learning rate, batch size, epochs, and output directory inside `TrainingArguments`

3. Execute the notebook cells to:

   * Load configuration from `config.json` and apply the DNABERT-2 tokenizer
   * Tokenize sequences and prepare Hugging Face `Dataset` objects
   * Instantiate `AutoModelForSequenceClassification` with three labels
   * Train with `Trainer`, periodically evaluating on the validation set
   * Save checkpoints and logs, including:

     * `model.safetensors`
     * `optimizer.pt`
     * `scheduler.pt`
     * `training_args.bin`
     * `trainer_state.json`
     * `baseline_metrics.json`
     * `rng_state.pth`

---

## Checkpoints and reproducibility

Reproduction and continuation of the training run rely on the following files:

* `config.json` and `configuration_bert.py`
  Define the architecture and configuration used by the model.

* `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json`
  Specify the exact tokenizer used for converting nucleotide sequences into token IDs.

* `model.safetensors`
  Contains the learned weights of the fine tuned model.

* `optimizer.pt` and `scheduler.pt`
  Store optimizer and scheduler states for exact training continuation.

* `training_args.bin`
  Encodes the original `TrainingArguments` object.

* `trainer_state.json`
  Provides a textual log of training progress and evaluation checkpoints.

* `rng_state.pth`
  Stores the random generator state for bit-wise reproducibility in the same environment.

With consistent software versions and hardware type, these artifacts allow very close reproduction of the training trajectory or direct reuse of the trained model for inference.

---

## Extending the project

Potential extensions for future work include:

* Increasing label granularity to distinguish individual mitochondrial genes, such as ND1, CO1, and CYB, and subregions within rRNA or tRNA genes.
* Adding tasks such as gene boundary detection, promoter and regulatory site annotation, or variant effect prediction, potentially in a multi-task learning setting with shared DNABERT-2 backbone.
* Incorporating cross-species mitochondrial genomes to test generalization and exploit evolutionary conservation across taxa.
* Benchmarking DNABERT-2 against other genomic foundation models and against simpler sequence models such as CNNs or k-mer based models.
* Adding calibration analysis, uncertainty estimation, and threshold tuning for clinical or diagnostic pipelines.

---

## Citation

For academic use, appropriate references may include:

* The DNABERT-2 paper that introduces the underlying foundation model.
* A MitoGPT report or manuscript describing the broader context of mitochondrial modeling, once available.

When a canonical citation format is finalized, a BibTeX entry can be added to this section.

---