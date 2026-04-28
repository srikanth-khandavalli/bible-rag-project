import os
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load Environment & API Key
print("\n[STEP 1/6] Loading system environment variables...")
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 2. Initialize the "Librarian" (Retriever)
print("\n[STEP 2/6] Initializing Local Embedding Model (Hugging Face)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("[STEP 3/6] Connecting to local ChromaDB 'Bible Brain'...")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
# # Initial retrieval of top 3 most relevant "seed" verses
# retriever = vector_db.as_retriever(search_kwargs={"k": 10})

# MMR will purposely look for "diverse" results, helping find both Johns
retriever = vector_db.as_retriever(
    search_type="mmr",
    # search_kwargs={'k': 20, 'fetch_k': 200, 'lambda_mult': 0.5}
    search_kwargs={'k': 50, 'fetch_k': 200, 'lambda_mult': 0.3}
)

# 3. Initialize the "Scholar" (Gemini LLM)
print("\n[STEP 4/6] Connecting to Gemini 2.5 Flash...")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

# 4. ADVANCED LOGIC: The Context Windowing Function
def get_context_with_window(retrieved_docs):
    print(f"\n🪟 WINDOWING: Expanding context for {len(retrieved_docs)} verses...")
    full_context_blocks = []
    
    for doc in retrieved_docs:
        # Get the current verse ID (e.g., 40001001 for Matt 1:1)
        # We ensure it's an integer for math, then a string for the ChromaDB lookup
        try:
            current_id = int(doc.metadata['id'])
            citation = doc.metadata.get('citation', 'Unknown Source')
            
            # Define window: 2 verses before to 2 verses after
            window_ids = [str(i) for i in range(current_id - 2, current_id + 3)]
            
            # Fetch the actual text for these IDs from the database
            window_data = vector_db.get(ids=window_ids)
            print(f"   -> Retrieved window for {citation}: IDs {window_ids}")
            window_verses = window_data['documents']
            # print(f"   -> Window verses for {citation}: {window_verses}")
            
            # Narrative blocks help the LLM see the 'bridge' between verses
            combined_narrative = " ".join(window_verses)
            print(f"   -> Expanded {citation} into a 5-verse narrative window.")
            full_context_blocks.append(f"[{citation} Narrative Context]: {combined_narrative}")
        except Exception as e:
            print(f"   -> Warning: Could not expand window for {doc.metadata.get('citation')}: {e}")
            full_context_blocks.append(f"[{doc.metadata.get('citation')}]: {doc.page_content}")
        # print(f"   -> Current full context block: {full_context_blocks}")    
    return "\n\n".join(full_context_blocks)

# 5. Defining the RAG Chain
print("[STEP 5/6] Configuring 'Biblical Scholar' prompt with Context Windowing...")
template = """You are a professional Biblical Scholar. Answer the question based ONLY on the following retrieved narrative context. 
Every block provided contains a 'Narrative Context' which includes the verses surrounding a specific reference to help you see the full story.

For every claim you make, you MUST cite the book, chapter, and verse from the context.
If the answer is not in the context, politely say that you cannot find a biblical reference for that topic.

Context:
{context}

Question: 
{question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# The chain now includes the windowing function before sending to the prompt
rag_chain = (
    {"context": retriever | get_context_with_window, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Execute the User Query
if __name__ == "__main__":
    user_query = "List the different Johns mentioned in the Bible and what they are known for."
    print(f"\n[STEP 6/6] PROCESSING USER QUERY: '{user_query}'")
    
    start_time = time.time()
    response = rag_chain.invoke(user_query)
    end_time = time.time()
    
    print("\n--- 📖 BIBLE STUDY ASSISTANT RESPONSE ---")
    print(response)
    print(f"\n(Processing took {end_time - start_time:.2f} seconds)")