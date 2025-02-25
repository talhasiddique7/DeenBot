import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def clean_quran_data(txt_path):
    # Read the merged Quran text file
    df = pd.read_csv(txt_path, delimiter="|", names=["Surah", "Ayah", "Arabic", "Urdu"], encoding="utf-8")
    
    # Create a unified text format (Arabic + Urdu translation)
    df["text"] = df.apply(lambda row: f"Ayah {row['Surah']}:{row['Ayah']}\nArabic: {row['Arabic']}\nUrdu: {row['Urdu']}", axis=1)
    
    return df

# Load and clean Quran data
quran_path = "quran_merged.txt"
df = clean_quran_data(quran_path)

# Load data into LangChain
loader = DataFrameLoader(df, page_content_column="text")
docs = loader.load()

# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
documents = text_splitter.split_documents(docs)

# Load optimized embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# Create FAISS vector store
db = FAISS.from_documents(documents, embeddings)

# Save FAISS index locally
db.save_local("vectorstore/quran_faiss")

print("✅ Quran FAISS index creation complete.")
