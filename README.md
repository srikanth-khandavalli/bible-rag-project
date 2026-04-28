# Bible Study RAG Assistant 📖🤖

An advanced Retrieval-Augmented Generation (RAG) system designed to assist in deep biblical study. This tool bridges the gap between static scripture and AI reasoning.

## 🌟 Key Features
- **Dual Brain Architecture**: Toggle between Cloud (Gemini) for speed and Local (Llama 3.2) for 100% privacy.
- **Narrative Windowing**: Don't just get verses; get the story. The system automatically retrieves surrounding context (±2 verses).
- **Theological Disambiguation**: Specifically engineered to distinguish between biblical figures with the same name (e.g., the multiple Johns).
- **KJV Optimized**: Uses a specialized query expansion layer to translate modern questions into KJV-friendly search terms.

## 🛠️ Tech Stack
- **Orchestration**: LangChain
- **Vector DB**: ChromaDB
- **Embeddings**: Hugging Face `all-MiniLM-L6-v2` (Local)
- **Models**: Google Gemini 1.5 Flash & Meta Llama 3.2 (via Ollama)
## Installation Commands
1. Windows (PowerShell)On Windows, you can use a one-line command to download and run the installer automatically:
    Win+R --> cmd
    irm https://ollama.com/install.ps1 | iex

2. Alternatively, you can download the .exe manually from ollama.com.
Linux (Terminal)Linux uses a simple curl script that handles the entire setup, including GPU support detection:

    Bashcurl -fsSL https://ollama.com/install.sh | sh

2. Pulling Llama 3.2


    ollama pull llama3.2

3. VerificationAfter the download finishes, verify it's in your local "library" by running:

    ollama list

## 🚀 Quick Start with local LLM
1. **Clone & Install**:
   ```bash
   pip install -r requirements.txt

2. **Setup Local LLM (Windows)**:
    ```bash
    irm https://ollama.com/install.ps1 | iex
    ollama pull llama3.2
2. **Setup Local LLM (Linux)**:
    ```bash
    Bashcurl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3.2

3. **Ingest Data**:
    ```bash
    python src/ingestion.py

4. **Run Study**:
    ```bash
    python src/local_rag.py

