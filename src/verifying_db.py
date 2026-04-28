from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load the existing brain using the updated packages
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Test search
query = "Who is John?"
docs = vector_db.similarity_search(query, k=3)
print(f"Top 3 results for query: '{query}'")
for i, doc in enumerate(docs, 1):
    print(f"{i}. {doc.metadata['citation']} - {doc.page_content[:100]}...") 
# print(f"✅ Top result found: {docs[0].metadata['citation']}")
# print(f"Text: {docs[0].page_content}")