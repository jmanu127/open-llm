# open-llm
Building an llm from scratch

llm-project/
│
├── configs/                # YAML/JSON configs (model, training, data)
│   ├── model/
│   ├── training/
│   └── data/
│
├── data/
│   ├── raw/                # original datasets
│   ├── processed/          # cleaned + tokenized
│   └── shards/             # training-ready chunks
│
├── src/
│   ├── data/               # data processing pipeline
│   │   ├── loaders.py
│   │   ├── cleaning.py
│   │   ├── tokenizer.py
│   │   └── dataset.py
│   │
│   ├── model/              # model architecture
│   │   ├── transformer.py
│   │   ├── attention.py
│   │   ├── embeddings.py
│   │   └── config.py
│   │
│   ├── training/           # training logic
│   │   ├── trainer.py
│   │   ├── optimizer.py
│   │   ├── scheduler.py
│   │   └── checkpoint.py
│   │
│   ├── evaluation/         # eval + benchmarks
│   │   ├── metrics.py
│   │   ├── benchmarks.py
│   │   └── eval_runner.py
│   │
│   ├── inference/          # generation / serving
│   │   ├── generate.py
│   │   ├── sampling.py
│   │   └── api.py
│   │
│   ├── utils/              # shared utilities
│   │   ├── logging.py
│   │   ├── seed.py
│   │   └── distributed.py
│   │
│   └── main.py             # entry point
│
├── scripts/                # CLI scripts
│   ├── train.py
│   ├── preprocess.py
│   ├── evaluate.py
│   └── serve.py
│
├── tests/                  # unit + integration tests
│
├── experiments/            # experiment tracking outputs
│
├── checkpoints/            # saved models
│
├── notebooks/              # research / exploration
│
├── requirements.txt / pyproject.toml
├── README.md
└── .gitignore
