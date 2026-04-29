import os
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import google.genai.errors as genai_errors 

# ==========================================
# 1. SETUP & ENVIRONMENT
# ==========================================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print("\n[CLOUD] Initializing Librarian (Hugging Face) & Bible Brain (ChromaDB)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

print("[CLOUD] Awakening Gemini 1.5 Flash...")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def expand_query(user_query):
    """Keep search queries simple so they match the KJV text style."""
    print(f"\n🔍 [CLOUD] ANALYZING QUERY: '{user_query}'")
    prompt = f"""You are a Biblical Scholar. The user asked: '{user_query}'
    Break this into 3 simple search phrases that would appear in the Bible.
    Avoid modern words like 'account' or 'identity'.
    Example for John: 'John the Baptist', 'John son of Zebedee', 'John Mark'
    Output ONLY the phrases, one per line."""
    
    response = llm.invoke(prompt).content
    queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
    print(f"   -> Sub-queries for Librarian: {queries}")
    return queries

def get_context_with_window(queries, window_size=2):
    """Retrieves 'seed' verses and expands them into narrative windows, returning text and metadata."""
    seen_ids = set()
    full_context_blocks = []
    source_metadata_list = [] # Holds the data for the UI cards

    print(f"🪟 [CLOUD] WINDOWING: Searching for {len(queries)} sub-queries...")
    
    for q in queries:
        results = vector_db.similarity_search(q, k=3)
        
        for doc in results:
            v_id_str = doc.metadata.get('id')
            if not v_id_str or v_id_str in seen_ids:
                continue
            
            seen_ids.add(v_id_str)
            v_id = int(v_id_str)
            citation = doc.metadata.get('citation', 'Unknown')
            genre = doc.metadata.get('genre', 'Unknown')
            testament = doc.metadata.get('testament', 'Unknown')

            # Windowing Math
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
    return context_string, source_metadata_list

# ==========================================
# 3. THE MAIN CLOUD RAG LOGIC
# ==========================================
def run_bible_study(user_query, k=5, window=2):
    # Phase A: Expansion
    try:
        sub_queries = expand_query(user_query)
    except Exception as e:
        print(f"⚠️ Cloud expansion failed: {e}")
        sub_queries = [user_query]
        
    # Phase B: Retrieval & Windowing
    context, sources = get_context_with_window(sub_queries, window_size=window)
    
    if not context:
        return "I'm sorry, I couldn't find any relevant verses.", []

    # Phase C: Final Answer Generation 
    print("\n✍️ [CLOUD] SCHOLAR: Synthesizing final response...")
    template = """You are a professional Biblical Scholar. Answer using ONLY the context.
    Cite every claim. Context:\n{context}\n\nQuestion: {question}\nAnswer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    # Robust retry + fallback for cloud rate limits (503 errors)
    for attempt in range(3):
        try:
            response = chain.invoke({"context": context, "question": user_query})
            # CRITICAL FIX: Return tuple for the UI
            return response, sources
        except Exception as e:
            print(f"⚠️ LLM Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2) 
            else:
                fallback_msg = (
                    "❌ NOTICE: The AI Scholar is currently overloaded (503 Service Unavailable).\n"
                    "While I cannot provide a summary, you can read the relevant verses below.\n"
                    "TIP: Try your question again in 30 seconds."
                )
                return fallback_msg, sources

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    query = "What happened at the Mount of Olives?"
    start_time = time.time()
    
    result_text, result_sources = run_bible_study(query, window=2)
    
    print("\n" + "="*50)
    print("📖 CLOUD BIBLE STUDY ASSISTANT (GEMINI 1.5)")
    print("="*50)
    print(result_text)
    print("="*50)
    print(f"Retrieved {len(result_sources)} source narratives.")
    print(f"Cloud processing took {time.time() - start_time:.2f} seconds")