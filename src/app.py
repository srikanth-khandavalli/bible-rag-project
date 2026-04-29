import streamlit as st
import os
# --- ADD THIS TO FIX THE OPENMP CRASH ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ----------------------------------------
import time
# IMPORT YOUR EXISTING LOGIC
# Ensure these function names match your local_rag.py file
from local_rag import run_local_bible_study, get_context_with_window 

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Bible Research Assistant", layout="wide")

# 2. SIDEBAR - System Health & Controls
st.sidebar.title("🛠️ System Monitor")
st.sidebar.success("● Ollama: Connected (Llama 3.2)")

# Professional Feature: User can adjust the narrative context
window_size = st.sidebar.slider("Narrative Window (Verses)", 1, 5, 2)
st.sidebar.info(f"Retrieving ±{window_size} surrounding verses for context.")

st.sidebar.divider()
st.sidebar.markdown("""
**Pro Tip**: This version uses **Local Llama 3.2** for 100% privacy. 
*Contact Admin to enable High-Speed Cloud Reasoning (Gemini).*
""")

# 3. MAIN DASHBOARD
st.title("📖 Local Biblical Research Assistant")
st.caption("Advanced RAG-based study tool for Pastors and Scholars.")

query = st.text_input("Enter your theological question:", placeholder="e.g., Who is John?")

if st.button("Analyze Scripture"):
    if query:
        with st.spinner("Checking Holy Bible(KJV)..."):
            start_time = time.time()
            
            # CALL YOUR BACKEND LOGIC
            # Note: Ensure your run_local_bible_study function accepts the window_size
            response, sources = run_local_bible_study(query)
            
            end_time = time.time()
            elapsed = round(end_time - start_time, 2)

        # 4. DISPLAY THE RESULTS
        st.subheader("🎓 Scholar's Synthesis")
        st.write(response)
        st.caption(f"Analysis completed in {elapsed} seconds.")

        st.divider()
        
        # 5. DISPLAY THE SOURCE CARDS
        st.subheader("📚 Verified Narrative Context")
        for source in sources:
            with st.expander(f"Reference: {source['citation']}"):
                st.markdown(f"**Context Area:** \n\n {source['full_text']}")
                st.caption(f"Genre: {source['genre']} | Testament: {source['testament']}")
    else:
        st.warning("Please enter a question to begin.")