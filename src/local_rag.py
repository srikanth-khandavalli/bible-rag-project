import os
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

# ==========================================
# 1. SETUP & ENVIRONMENT
# ==========================================
print("\n[LOCAL] Initializing Librarian (Hugging Face) & Bible Brain (ChromaDB)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

print("[LOCAL] Awakening Llama 3.2 (3B)...")
llm = OllamaLLM(model="llama3.2")


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def expand_query(user_query):
    """Uses the LLM to turn 1 vague question into simple KJV-friendly search targets."""
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

def get_context_with_window(queries, window_size=2):
    """Retrieves 'seed' verses and expands them into narrative windows, returning text and metadata."""
    seen_ids = set()
    full_context_blocks = []
    source_metadata_list = [] # Holds the data for the UI cards

    print(f"🪟 [LOCAL] WINDOWING: Searching for {len(queries)} sub-queries...")
    
    for q in queries:
        # Search the vector database
        results = vector_db.similarity_search(q, k=3)
        # ADD THIS DIAGNOSTIC LINE:
        print(f"RAW RESULTS FOR '{q}':", [doc.metadata for doc in results])
        
        for doc in results:
            v_id_str = doc.metadata.get('id')
            if not v_id_str or v_id_str in seen_ids:
                continue
            
            seen_ids.add(v_id_str)
            v_id = int(v_id_str)
            citation = doc.metadata.get('citation', 'Unknown')
            genre = doc.metadata.get('genre', 'Unknown')
            testament = doc.metadata.get('testament', 'Unknown')

            # Windowing Math using the slider variable from the UI
            window_ids = [str(i) for i in range(v_id - window_size, v_id + window_size + 1)]
            window_data = vector_db.get(ids=window_ids)
            verses = window_data.get('documents', [])
            
            if verses:
                narrative = " ".join(verses)
                full_context_blocks.append(f"[{citation} Narrative]: {narrative}")
                
                # Build the dictionary for Streamlit expanders
                source_metadata_list.append({
                    "citation": citation,
                    "full_text": narrative,
                    "genre": genre,
                    "testament": testament
                })
            
    context_string = "\n\n".join(full_context_blocks)
    
    # Return BOTH the string for the LLM and the list for the UI
    return context_string, source_metadata_list


# ==========================================
# 3. THE MAIN LOCAL RAG LOGIC
# ==========================================
def run_local_bible_study(user_query, k=5, window=2):
    """Main pipeline: Expands query, retrieves windowed context, and generates an answer."""
    # Phase A: Expansion
    try:
        sub_queries = expand_query(user_query)
    except Exception as e:
        print(f"⚠️ Local expansion failed: {e}")
        sub_queries = [user_query]
    
    # Phase B: Retrieval & Windowing (Local ChromaDB)
    context, sources = get_context_with_window(sub_queries, window_size=window)
    
    if not context:
        return "I'm sorry, I couldn't find any relevant verses locally.", []

    # Phase C: Final Answer Generation 
    print("\n✍️ [LOCAL] SCHOLAR: Synthesizing final response...")
    template = """You are a professional Biblical Scholar. Answer the question based ONLY on the following narrative context.
    Cite every claim (Book Chapter:Verse). Distinguish between people with the same name.

    Context:
    {context}

    Question: {question}

    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({"context": context, "question": user_query})
        # CRITICAL FIX: Return both the LLM response AND the sources list
        return response, sources
    except Exception as e:
        error_msg = (f"❌ LOCAL ERROR: Llama 3.2 failed to generate.\n\n"
                     f"📖 RAW VERSES RETRIEVED:\n{context}")
        return error_msg, sources


# ==========================================
# 4. EXECUTION (For testing directly in terminal)
# ==========================================
if __name__ == "__main__":
    query = "Who is John?"
    start_time = time.time()
    
    # Unpack the tuple here as well for terminal testing
    result_text, result_sources = run_local_bible_study(query, window=2)
    
    print("\n" + "="*50)
    print("📖 LOCAL BIBLE STUDY ASSISTANT (LLAMA 3.2)")
    print("="*50)
    print(result_text)
    print("="*50)
    print(f"Retrieved {len(result_sources)} source narratives.")
    print(f"Local processing took {time.time() - start_time:.2f} seconds")