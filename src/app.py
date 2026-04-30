import streamlit as st
import time

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Bible Research Assistant", layout="wide")

# 2. SIDEBAR - System Health & Controls
st.sidebar.title("🛠️ System Controls")

# --- THE NEW TOGGLE SWITCH ---
st.sidebar.subheader("Engine Selection")
engine_choice = st.sidebar.radio(
    "Choose your AI Scholar:",
    ["Local Llama (Privacy & Secure)", "Cloud Gemini (High-Speed)"],
    help="Local is 100% private but slower. Cloud is fast but uses an external API."
)

# Visual indicator of which engine is active
if "Local" in engine_choice:
    st.sidebar.success("● Engine: Llama (Local)")
else:
    st.sidebar.success("● Engine: Gemini (Cloud)")

st.sidebar.divider()

# Narrative Window Slider
window_size = st.sidebar.slider("Verses Search Window", 1, 5, 2)
st.sidebar.info(f"Retrieving ±{window_size} surrounding verses for context.")

st.sidebar.divider()
st.sidebar.markdown("""
**Pro Tip**: Use Local Mode for sensitive studies and Cloud Mode for rapid sermon prep.
""")

# 3. MAIN DASHBOARD
st.title("📖 Biblical Research Assistant")
st.caption("Advanced RAG-based study tool for Pastors and Scholars.")

query = st.text_input("Enter your theological question:", placeholder="e.g., What is the purpose of life?")

if st.button("Analyze Scripture"):
    if query:
        with st.spinner(f"Running {engine_choice.split()[0]} engine..."):
            start_time = time.time()
            
            # --- ROUTING LOGIC ---
            # We import the required function dynamically based on the toggle switch
            if "Local" in engine_choice:
                from local_rag import run_local_bible_study
                # Make sure run_local_bible_study accepts the window parameter!
                response, sources = run_local_bible_study(query, window=window_size)
            else:
                from rag_engine import run_bible_study
                # Make sure run_bible_study accepts the window parameter!
                response, sources = run_bible_study(query, window=window_size)
            
            end_time = time.time()
            elapsed = round(end_time - start_time, 2)

        # 4. DISPLAY THE RESULTS
        st.subheader("🎓 AI Analysis")
        st.write(response)
        st.caption(f"Analysis completed in {elapsed} seconds.")

        st.divider()
        
        # 5. DISPLAY THE SOURCE CARDS
        st.subheader("📚 Scriptures referred")
        
        if sources:
            for source in sources:
                with st.expander(f"Reference: {source.get('citation', 'Unknown')}"):
                    st.markdown(f"**Context Area:** \n\n {source.get('full_text', 'No text available.')}")
                    st.caption(f"Genre: {source.get('genre', 'N/A')} | Testament: {source.get('testament', 'N/A')}")
        else:
            st.info("No contextual sources were returned for this query.")
    else:
        st.warning("Please enter a question to begin.")