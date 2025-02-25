# DeenBot RAG Chatbot

A lightweight **Retrieval-Augmented Generation (RAG) chatbot** built using **LangChain**, **FAISS**, and **sentence-transformers** to answer queries based on a Quranic text knowledge base. Designed for **efficient local use within 8GB RAM**.

## 🚀 Features
- **RAG-based chatbot** using FAISS for fast retrieval.
- **Uses a structured text file** as a knowledge base.
- **Runs locally on CPU** with optimized embeddings.
- **Efficient embedding model** for Quranic text.
- **Streamlit UI** for an interactive experience.

## 🛠 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/talhasiddique7/DeenBot.git
   cd DeenBot
   ```

2. Install Python 3.10 (if not installed):
   ```bash
   sudo apt update
   sudo apt install python3.10 python3.10-venv python3.10-dev
   ```

3. Create a virtual environment and activate it:
   ```bash
   python3.10 -m venv rag_env
   source rag_env/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📥 Preparing the Data

Place your **Quranic text file** (e.g., `quran_merged.txt`) in the root directory.

Run the following command to **embed the data and create a FAISS index**:
   ```bash
   python ingest.py
   ```

## 🏃 Running the Chatbot

Start the chatbot with:
   ```bash
   streamlit run app.py
   ```

## ⚡ Performance Optimization Notes  

For **faster responses** (within 3 seconds), consider the following:  

1. **Use a smaller embedding model**:  
   - Instead of large models like `all-MiniLM-L12-v2`, use **`all-MiniLM-L6-v2`** for better speed.  

2. **Reduce max tokens**:  
   - Set `max_new_tokens=50` or lower to minimize generation time.  

3. **Adjust temperature and top-p**:  
   - Use `temperature=0.3` for deterministic outputs.  
   - Set `top_p=0.9` to prioritize high-probability responses.  

4. **Limit retrieval results**:  
   - Adjust FAISS retriever parameters:  
     ```python
     retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 2, 'fetch_k': 5})
     ```
   - This reduces the number of retrieved documents, speeding up processing.  

5. **Optimize local execution**:  
   - Run on a machine with **at least 8GB RAM** for smooth performance.  
   - Use **GPU acceleration** if available (via `llama-cpp-python` or `CTransformers`).  

By applying these optimizations, your **Quran RAG chatbot** will respond in under **3 seconds**. 🚀  

## 📌 Example Queries
- *"What is the translation of Surah Al-Fatiha?"*
- *"Explain Ayah 2:255"*
- *"Find similar Ayahs on patience"*

## 💡 Technologies Used
- **LangChain** for RAG processing.
- **FAISS** for efficient vector search.
- **Sentence-Transformers** for embeddings.
- **Streamlit** for an interactive UI.

## 👤 Author
**Talha Siddique**

---

💬 *Feel free to contribute and improve this project!*
