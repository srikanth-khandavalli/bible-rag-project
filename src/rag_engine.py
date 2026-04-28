import os
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import google.genai.errors as genai_errors 

# SETUP
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

def expand_query(user_query):
    """Keep search queries simple so they match the KJV text style."""
    print(f"\n🔍 ANALYZING QUERY: '{user_query}'")
    prompt = f"""You are a Biblical Scholar. The user asked: '{user_query}'
    Break this into 3 simple search phrases that would appear in the Bible.
    Avoid modern words like 'account' or 'identity'.
    Example for John: 'John the Baptist', 'John son of Zebedee', 'John Mark'
    Output ONLY the phrases, one per line."""
    
    response = llm.invoke(prompt).content
    queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
    print(f"   -> Librarian will search for: {queries}")
    return queries

def get_context_with_window(queries):
    seen_ids = set()
    full_context_blocks = []

    for q in queries:
        # Use simple similarity search
        results = vector_db.similarity_search(q, k=3)
        
        for doc in results:
            v_id_str = doc.metadata.get('id')
            if not v_id_str or v_id_str in seen_ids:
                continue
            
            seen_ids.add(v_id_str)
            v_id = int(v_id_str)
            citation = doc.metadata.get('citation', 'Unknown')

            # Windowing: 2 before, 2 after
            window_ids = [str(i) for i in range(v_id - 2, v_id + 3)]
            window_data = vector_db.get(ids=window_ids)
            verses = window_data.get('documents', [])
            
            if verses:
                narrative = " ".join(verses)
                full_context_blocks.append(f"[{citation} Narrative]: {narrative}")

    return "\n\n".join(full_context_blocks)

def run_bible_study(query):
    # Phase A: Expansion (First potential failure point)
    try:
        sub_queries = expand_query(query)
    except Exception as e:
        print(f"⚠️ Query Expansion failed: {e}. Falling back to original query.")
        sub_queries = [query] # Just use the original query if expansion fails

    # Phase B: Retrieval & Windowing (Local - should never fail)
    context = get_context_with_window(sub_queries)
    
    if not context:
        return "I'm sorry, I couldn't find any relevant verses in the database."

    # Phase C: Final Answer Generation (Second potential failure point)
    print("\n✍️ SCHOLAR: Attempting synthesis...")
    template = """You are a professional Biblical Scholar. Answer using ONLY the context.
    Cite every claim. Context:\n{context}\n\nQuestion: {question}\nAnswer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    # We wrap the final generation in a robust retry + fallback
    for attempt in range(3):
        try:
            return chain.invoke({"context": context, "question": query})
        except Exception as e:
            # We check if it's a 503 or other server error
            print(f"⚠️ LLM Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2) # Wait a bit before retrying
            else:
                # This is the "Fallback" you wanted
                return (
                    "❌ NOTICE: The AI Scholar is currently overloaded (503 Service Unavailable).\n"
                    "While I cannot provide a summary, I have successfully retrieved the "
                    "relevant verses for your study below:\n\n"
                    "📖 RETRIEVED BIBLE CONTEXT:\n"
                    "===========================================================\n"
                    f"{context}\n"
                    "===========================================================\n"
                    "TIP: Try your question again in 30 seconds."
                )

if __name__ == "__main__":
    print(run_bible_study("What happened at the Mount of Olives?"))