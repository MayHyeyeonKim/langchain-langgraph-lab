# LoRA learning experiment

Do not start fine-tuning until the base workflow has repeatable tests and review examples.

1. Collect small JSONL examples of `document`, `analysis`, `revision`, and `review`.
2. Split examples into train and held-out evaluation sets before training.
3. Start with a 0.5B–1.5B model on Apple Silicon/MPS; do not attempt full 8B training on 16GB RAM.
4. Train LoRA adapters, compare them with the base model on the held-out set, and only then explore a compatible Ollama export.

The optional `training` dependency group deliberately stays separate from the application runtime.

## First safe exercise

```bash
uv run python training/prepare_dataset.py training/example_dataset.jsonl
```

Every row uses three non-empty strings: `instruction`, `input`, and `output`. Build a held-out
evaluation set before training; do not train and assess the model on the same examples.

## Starter train/eval split

The small included split demonstrates the correct separation. It is for learning the format only,
not sufficient for fine-tuning a useful model.

```bash
uv run python training/prepare_dataset.py training/data/train.jsonl
uv run python training/prepare_dataset.py training/data/eval.jsonl
```

Add real, reviewed RAG runs to `train.jsonl`. Keep different documents and scenarios in
`eval.jsonl` so it remains an honest held-out evaluation set.
