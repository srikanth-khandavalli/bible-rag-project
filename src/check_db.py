from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Connect to your existing brain
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 2. Peek at the raw data
# This returns the raw dictionary from the underlying collection
print("\n--- 🔍 PEERING INTO CHROMADB ---")
data = vector_db._collection.peek(limit=5)
print(data.keys())  # Should show 'ids', 'metadatas', 'documents', 'embeddings'
print(data['ids'])
for i in range(len(data['ids'])):
    print(f"\n[ENTRY {i+1}]")
    print(f"ID: {data['ids'][i]}")
    print(f"Metadata: {data['metadatas'][i]}")
    print(f"Snippet: {data['documents'][i][:50]}...")

# 3. Check specific IDs
# Let's see if the ID '42001063' (Luke 1:63) actually exists
test_ids = ['42001061', '42001062', '42001063', '42001064', '42001065']
# test_ids = ['1001005', '1001006', '1001007', '1001008', '1001009']  # Matt 1:5-9
print(f"\n--- 🎯 TESTING WINDOW IDs: {test_ids} ---")
found_data = vector_db._collection.get(ids=test_ids)
print(f"Documents found: {len(found_data['documents'])}")
print(f"Content: {found_data['documents']}")