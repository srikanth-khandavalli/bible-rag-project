import os
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

# 1. SETUP & ENVIRONMENT
print("\n[LOCAL] Initializing Librarian (Hugging Face) & Bible Brain (ChromaDB)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

print("[LOCAL] Awakening Phi-3 (Local Scholar)...")
# Ensure Ollama is running and you have run 'ollama pull llama3.1 in your terminal'
print("[LOCAL] Awakening Llama 3.2 (3B)...")
llm = OllamaLLM(model="llama3.2")

# 2. HELPER FUNCTIONS (Identical to rag_engine.py)

def expand_query(user_query):
    """Uses Phi-3 to turn 1 vague question into simple KJV-friendly search targets."""
    print(f"\n🔍 [LOCAL] ANALYZING QUERY: '{user_query}'")
    expansion_prompt = f"""
    You are a Biblical Research Assistant. The user asked: '{user_query}'
    Break this into 3 simple search phrases that would appear in the Bible.
    Avoid modern words. Example for John: 'John the Baptist', 'John son of Zebedee'
    Output ONLY the phrases, one per line."""
    
    response = llm.invoke(expansion_prompt)
    queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
    print(f"   -> Sub-queries for Librarian: {queries}")
    return queries

def get_context_with_window(queries):
    """Retrieves 'seed' verses and expands them into narrative windows."""
    seen_ids = set()
    full_context_blocks = []

    print(f"🪟 [LOCAL] WINDOWING: Searching for {len(queries)} sub-queries...")
    
    for q in queries:
        results = vector_db.similarity_search(q, k=3)
        
        for doc in results:
            v_id_str = doc.metadata.get('id')
            if not v_id_str or v_id_str in seen_ids:
                continue
            
            seen_ids.add(v_id_str)
            v_id = int(v_id_str)
            citation = doc.metadata.get('citation', 'Unknown')

            # Windowing Math: 2 before, 2 after
            window_ids = [str(i) for i in range(v_id - 2, v_id + 3)]
            window_data = vector_db.get(ids=window_ids)
            verses = window_data.get('documents', [])
            
            if verses:
                narrative = " ".join(verses)
                full_context_blocks.append(f"[{citation} Narrative]: {narrative}")
            
    return "\n\n".join(full_context_blocks)

# 3. THE MAIN LOCAL RAG LOGIC

def run_local_bible_study(user_query):
    # Phase A: Expansion (using local Phi-3)
    try:
        sub_queries = expand_query(user_query)
    except Exception as e:
        print(f"⚠️ Local expansion failed: {e}")
        sub_queries = [user_query]
    
    # Phase B: Retrieval & Windowing (Local ChromaDB)
    context = get_context_with_window(sub_queries)
    
    if not context:
        return "I'm sorry, I couldn't find any relevant verses locally."

    # Phase C: Final Answer Generation (using local Phi-3)
    print("\n✍️ [LOCAL] SCHOLAR: Synthesizing final response...")
    template = """You are a professional Biblical Scholar. Answer the question based ONLY on the following narrative context.
    Cite every claim (Book Chapter:Verse). Distinguish between people with the same name.

    Context:
    {context}

    Question: {question}

    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    # Note: Local LLMs don't usually throw 503 errors, 
    # but we keep the structure for consistency.
    try:
        return chain.invoke({"context": context, "question": user_query})
    except Exception as e:
        return (f"❌ LOCAL ERROR: Phi-3 failed to generate.\n\n"
                f"📖 RAW VERSES RETRIEVED:\n{context}")

# 4. EXECUTION
if __name__ == "__main__":
    query = "Who is John?"
    start_time = time.time()
    
    result = run_local_bible_study(query)
    
    print("\n" + "="*50)
    print("📖 LOCAL BIBLE STUDY ASSISTANT (PHI-3)")
    print("="*50)
    print(result)
    print("="*50)
    print(f"Local processing took {time.time() - start_time:.2f} seconds")