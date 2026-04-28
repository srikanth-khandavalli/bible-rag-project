import pandas as pd
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 1. Load the CSV Data
print("Loading Bible data...")
df_text = pd.read_csv('data/raw/t_kjv.csv')
df_key = pd.read_csv('data/raw/key_english.csv')

# Clean up column names
df_key.rename(columns={'t': 'testament'}, inplace=True)
df_text.rename(columns={'t': 'scripture_text'}, inplace=True)
bible_df = pd.merge(df_text, df_key[['b', 'n', 'testament']], on='b')

# 2. Create Documents
documents = []
for _, row in bible_df.iterrows():
    citation = f"{row['n']} {row['c']}:{row['v']}"
    doc = Document(
        page_content=row['scripture_text'],
        metadata={
            "id": str(row['id']),
            "citation": citation,
            "book": row['n']
        }
    )
    documents.append(doc)

# 3. Initialize Local Embeddings
print("Initializing local embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Create and Persist the Vector Store
print(f"Indexing {len(documents)} verses with explicit IDs...")
all_ids = [doc.metadata["id"] for doc in documents]

# DELETE your 'chroma_db' folder before running this!
vector_db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    ids=all_ids,
    persist_directory="./chroma_db"
)

print("✅ Ingestion Complete. 'Bible Brain' is ready.")