
# 📖 Local Biblical Research Assistant (RAG)

![Python Version](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Orchestration-1C3C3C)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-white?logo=ollama)
![Gemini](https://img.shields.io/badge/Google_Gemini-Cloud_LLM-4285F4)

An advanced Retrieval-Augmented Generation (RAG) application designed for Pastors, Theologians, and Biblical Scholars. This tool allows users to ask complex theological questions and receive synthesized answers based **strictly** on verified biblical narratives, with full control over data privacy.

---

## ✨ Key Features

* **Dual-Engine Architecture (Privacy vs. Speed):**
  * 🔒 **Local Mode (Llama 3.2):** 100% offline, private, and secure using Ollama. Perfect for sensitive pastoral research.
  * ⚡ **Cloud Mode (Gemini 1.5 Flash):** High-speed, highly formatted reasoning via Google's API for rapid sermon prep.
* **Dynamic Narrative Windowing:** Does not just retrieve single verses. It retrieves "seed" verses and automatically pulls surrounding verses (±2 to ±5) to provide complete literary and historical context.
* **Source Verification:** Every synthesis includes expandable UI cards displaying the exact referenced verses, citations, and metadata (Genre, Testament).
* **Remote Access Ready:** Includes a batch script for securely tunneling the local server to the public internet via Ngrok for remote presentations and sharing.

## 🛠️ Technology Stack

* **Frontend UI:** Streamlit
* **RAG Orchestration:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings Model:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **Generative Models:** Ollama (Llama 3.2 3B), Google Gemini 1.5 Flash

---

## 📋 Prerequisites

Before running this project, ensure you have the following installed:
1. **Miniconda / Anaconda:** For managing the Python environment.
2. **Ollama:** Download from [ollama.com](https://ollama.com/). Once installed, open your terminal and pull the required model:
   ```bash
   ollama pull llama3.2
   ```
3. **Ngrok (Optional):** For remote access tunneling.

---

## 🚀 Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
```

**2. Create and activate the environment**
```bash
conda create -n bible_study_rag python=3.11 -y
conda activate bible_study_rag
```

**3. Install dependencies**
```bash
python -m pip install -r requirements.txt
```

**4. Configure Environment Variables**
Create a `.env` file in the root directory and add your Google Gemini API key (required for Cloud Mode):
```env
GEMINI_API_KEY="your_actual_api_key_here"
```

**5. Ingest the Data (First Time Only)**
*Note: The vector database is not included in source control due to size constraints.*
Place your raw Bible data file in the `src/` folder and run the ingestion script to build your local ChromaDB:
```bash
python src/ingest.py
```

---

## 🏃‍♂️ Running the Application

### Option A: Standard Local Run
To run the dashboard locally on your machine:
```bash
streamlit run src/app.py
```

### Option B: Remote Tunnel Run (Always-On)
If you have configured Ngrok for a static domain, simply run the master batch script. This will initialize Ollama, boot Streamlit, and open your secure public URL simultaneously.
```bash
Start_Server.bat
```

---

## 📂 Project Structure

```text
bible-RAG-project/
│
├── src/
│   ├── app.py                # Main Streamlit dashboard UI
│   ├── local_rag.py          # Llama 3.2 / Ollama RAG pipeline
│   ├── rag_engine.py         # Gemini 1.5 / Cloud RAG pipeline
│   ├── ingest.py             # Script to convert text to embeddings
│   └── chroma_db/            # Local vector database (Git Ignored)
│
├── .env                      # API keys (Git Ignored)
├── .gitignore                # Excludes DB and Cache files
├── requirements.txt          # Python dependencies
├── Start_Server.bat          # Master startup script for Ngrok
└── README.md                 # Project documentation
```

---

## 🐛 Troubleshooting

**Windows OpenMP Error (`libiomp5md.dll`)**
If the app crashes on launch with an OpenMP multiple-runtime conflict, run this command in your active Conda environment to bypass the duplicate library error:
```bash
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE
```
*(You must deactivate and reactivate the conda environment for this to take effect).*

# Additional Notes

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

