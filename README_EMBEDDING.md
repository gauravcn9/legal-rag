# Embeddings + FAISS (BAAI/bge-base-en-v1.5)

This project adds `create_embeddings.py` which:
- Loads `BAAI/bge-base-en-v1.5` via `sentence-transformers`.
- Splits input text into chunks, computes embeddings, and builds a FAISS index.

Install dependencies:

```bash
pip install -r requirements.txt
```

Create embeddings and index for a single file:

```bash
python create_embeddings.py --input legal_docs/mydoc.txt --output-dir faiss_store
```

Create embeddings for all `.txt`, `.md`, and `.pdf` files under a directory:

```bash
python create_embeddings.py --input legal_docs --output-dir faiss_store --extensions .txt,.md,.pdf
```

Quick search example (Python):

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

model = SentenceTransformer('BAAI/bge-base-en-v1.5')
q = 'Your query here'
q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype('float32')

index = faiss.read_index('faiss_store/faiss_index.index')
D, I = index.search(q_emb, 5)
meta = json.load(open('faiss_store/index_metadata.json'))
for idx in I[0]:
    if idx == -1:
        continue
    print(idx, meta['metadatas'][idx]['source'], meta['metadatas'][idx]['chunk_index'])
```

Notes:
- The first run will download the model weights from Hugging Face.
- The script normalizes embeddings and uses inner-product search (cosine similarity).
- If you have an existing FAISS index you can replace `IndexFlatIP` with a more efficient index (HNSW, IVF, etc.).
