# LoRA learning experiment

Do not start fine-tuning until the base workflow has repeatable tests and review examples.

1. Collect small JSONL examples of `document`, `analysis`, `revision`, and `review`.
2. Split examples into train and held-out evaluation sets before training.
3. Start with a 0.5B–1.5B model on Apple Silicon/MPS; do not attempt full 8B training on 16GB RAM.
4. Train LoRA adapters, compare them with the base model on the held-out set, and only then explore a compatible Ollama export.

The optional `training` dependency group deliberately stays separate from the application runtime.
