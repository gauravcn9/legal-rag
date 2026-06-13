#!/usr/bin/env python3
"""
create_embeddings.py
Create embeddings for text files using BAAI/bge-base-en-v1.5 and save a FAISS index.

Usage:
  python create_embeddings.py --input path/to/file_or_dir --output-dir faiss_store
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict]:
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be > chunk_overlap")
    chunks = []
    pos = 0
    n = len(text)
    idx = 0
    while pos < n:
        end = min(pos + chunk_size, n)
        chunk = text[pos:end].strip()
        if chunk:
            chunks.append({"text": chunk, "start": pos, "end": end, "chunk_index": idx})
            idx += 1
        pos += chunk_size - chunk_overlap
    return chunks


def extract_text_from_pdf(path: str) -> str:
    try:
        import fitz
        doc = fitz.open(path)
        pages = [page.get_text() for page in doc]
        return "\n".join(pages)
    except Exception:
        try:
            import PyPDF2
            from pathlib import Path
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
        except Exception as e:
            raise RuntimeError(f"Could not extract PDF text from {path}: {e}")

def read_files(input_path: str, extensions: List[str]) -> List[Dict]:
    p = Path(input_path)
    files = []
    if p.is_file():
        files = [p]
    else:
        for ext in extensions:
            for f in p.rglob(f"*{ext}"):
                files.append(f)

    docs = []
    for f in files:
        if f.suffix.lower() == '.pdf':
            text = extract_text_from_pdf(str(f))
        else:
            try:
                text = f.read_text(encoding="utf-8")
            except Exception:
                text = f.read_text(encoding="latin-1")
        docs.append({"path": str(f), "text": text})
    return docs


def main():
    parser = argparse.ArgumentParser(description="Create embeddings and FAISS index from text files.")
    parser.add_argument("--input", "-i", required=True, help="File or directory to embed")
    parser.add_argument("--output-dir", "-o", default="faiss_store", help="Output directory for index and metadata")
    parser.add_argument("--model", "-m", default="BAAI/bge-base-en-v1.5", help="Hugging Face model name")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--extensions", default='.txt,.md', help='Comma-separated list of extensions when input is a directory')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model} ...")
    model = SentenceTransformer(args.model)

    exts = [e.strip() if e.strip().startswith('.') else '.' + e.strip() for e in args.extensions.split(',')] if args.extensions else ['.txt', '.md']
    print("Reading files...")
    docs = read_files(args.input, extensions=exts)
    if not docs:
        print("No files found for the given input and extensions.")
        return

    all_chunks = []
    metadata = []
    for doc in docs:
        chunks = chunk_text(doc['text'], chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        for c in chunks:
            all_chunks.append(c['text'])
            metadata.append({"source": doc['path'], "start": c['start'], "end": c['end'], "chunk_index": c['chunk_index']})

    print(f"Total chunks: {len(all_chunks)}")
    if len(all_chunks) == 0:
        print("No text chunks to embed.")
        return

    print("Computing embeddings (this may download the model)...")
    embeddings = model.encode(all_chunks, batch_size=args.batch_size, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype='float32')

    dim = embeddings.shape[1]
    print(f"Embeddings shape: {embeddings.shape}")

    index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(index)

    ids = np.arange(0, embeddings.shape[0], dtype='int64')
    index.add_with_ids(embeddings, ids)

    faiss_index_path = out_dir / 'faiss_index.index'
    print(f"Saving FAISS index to {faiss_index_path}")
    faiss.write_index(index, str(faiss_index_path))

    meta = {"ids": ids.tolist(), "metadatas": metadata, "model": args.model}
    meta_path = out_dir / 'index_metadata.json'
    print(f"Saving metadata to {meta_path}")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    chunks_path = out_dir / 'chunks.jsonl'
    print(f"Saving chunks to {chunks_path}")
    with open(chunks_path, 'w', encoding='utf-8') as f:
        for i, (md, chunk) in enumerate(zip(metadata, all_chunks)):
            rec = {"id": int(i), "source": md['source'], "start": md['start'], "end": md['end'], "chunk_index": md['chunk_index'], "text": chunk}
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print("Done.")


if __name__ == '__main__':
    main()
