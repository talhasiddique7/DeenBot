#libraries import 
import streamlit as st
import time
from together import Together
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY")

# Streamlit UI
st.title("🔥 Chullah RAG Chatbot")
st.write("Ask me anything about Chullah!")

# Define Prompt Template to prevent hallucinations
custom_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="You are an AI assistant for Chullah. Answer **only** based on the following context:\n\n"
             "{context}\n\n"
             "If the question is unrelated to Chullah, simply say: "
             "'I can only answer questions related to Chullah.'\n\n"
             "Question: {query}\n"
             "Answer:"
)

@st.cache_resource
def load_faiss():
    """ Load FAISS database with embeddings """
    st.write("🔄 Loading FAISS vector database...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
        db = FAISS.load_local("vectorstore/db_faiss", embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 5, 'fetch_k': 5})
        st.success("✅ Database loaded successfully!")
        return retriever
    except Exception as e:
        st.error(f"❌ Error loading FAISS database: {e}")
        return None

@st.cache_resource
def load_together_ai():
    """ Initialize Together AI API client """
    st.write("🔄 Connecting to Together AI...")
    try:
        client = Together(api_key=TOGETHER_AI_API_KEY)
        st.success("✅ Together AI initialized!")
        return client
    except Exception as e:
        st.error(f"❌ Error initializing Together AI: {e}")
        return None

retriever = load_faiss()
together_client = load_together_ai()

def query_together_ai(context, question):
    """ Query the Together AI model with FAISS knowledge only """
    model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"  # Adjust as needed
    messages = [{"role": "user", "content": custom_prompt.format(context=context, query=question)}]

    try:
        response = together_client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

if retriever and together_client:
    query = st.text_input("Enter your query:")

    if st.button("Ask"):
        if query.strip():
            with st.spinner("⏳ Thinking..."):
                try:
                    start_time = time.time()  # Start timing
                    
                    # Retrieve relevant documents
                    docs = retriever.get_relevant_documents(query)
                    context = "\n".join([doc.page_content for doc in docs]) if docs else "No relevant data found."

                    # Ensure the model doesn't make up answers
                    if context == "No relevant data found.":
                        response_text = "⚠️ I can only answer questions related to Chullah."
                    else:
                        response_text = query_together_ai(context, query)

                    end_time = time.time()  # End timing
                    response_time = round(end_time - start_time, 2)  # Calculate response time

                    # Display response
                    st.subheader("**Answer:**")
                    st.write(response_text.strip())
                    st.write(f"⏱ Response Time: {response_time} seconds")

                    # Display sources
                    if docs:
                        st.subheader("📌 Sources:")
                        for i, doc in enumerate(docs, 1):
                            st.write(f"{i}. {doc.metadata.get('source', 'Unknown Source')}")
                    else:
                        st.info("No relevant sources found.")
                except Exception as e:
                    st.error(f"❌ Error processing query: {e}")
        else:
            st.warning("⚠️ Please enter a valid query.")
