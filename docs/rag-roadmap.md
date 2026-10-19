# RAG milestone

RAG lets the reviewer consult source material without retraining the model.

Use it only after the core reviewer works:

1. Extract and chunk reference documents with source and page metadata.
2. Store embeddings in a local Chroma collection.
3. Retrieve a small number of relevant chunks per review.
4. Pass the labelled context to Analyzer and Reviewer prompts.
5. Require outputs to identify the retrieved sources they relied on.

`document_reviewer.rag.build_context` is intentionally storage-agnostic so the retrieval backend can be learned independently.
