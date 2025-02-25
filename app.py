# Libraries Import
import streamlit as st
import time
from together import Together
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from deep_translator import GoogleTranslator
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY")

# Streamlit UI
st.title("📖 قرآن DeenBot")
st.write("قرآن کے بارے میں کوئی بھی سوال پوچھیں!")

# Define Prompt Template to prevent hallucinations
custom_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="آپ قرآن کی تشریح میں مہارت رکھنے والے ایک AI اسسٹنٹ ہیں۔ براہ کرم **صرف** درج ذیل سیاق و سباق کی بنیاد پر جواب دیں:\n\n"
             "{context}\n\n"
             "اگر سوال قرآن سے متعلق نہ ہو، تو بس کہیں: "
             "'میں صرف قرآن سے متعلق سوالات کے جوابات دے سکتا ہوں۔'\n\n"
             "سوال: {query}\n"
             "جواب:"
)

@st.cache_resource
def load_faiss():
    """ Load FAISS database with embeddings """
    st.write("🔄 FAISS وییکٹر ڈیٹا بیس لوڈ ہو رہا ہے...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local("vectorstore/quran_faiss", embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 3, 'fetch_k': 3})
        st.success("✅ ڈیٹا بیس کامیابی سے لوڈ ہو گیا!")
        return retriever
    except Exception as e:
        st.error(f"❌ FAISS ڈیٹا بیس لوڈ کرنے میں خرابی: {e}")
        return None

@st.cache_resource
def load_together_ai():
    """ Initialize Together AI API client """
    st.write("🔄 Together AI سے کنکشن قائم کیا جا رہا ہے...")
    try:
        client = Together(api_key=TOGETHER_AI_API_KEY)
        st.success("✅ Together AI کامیابی سے متحرک ہو گیا!")
        return client
    except Exception as e:
        st.error(f"❌ Together AI متحرک کرنے میں خرابی: {e}")
        return None

retriever = load_faiss()
together_client = load_together_ai()

def query_together_ai(context, question):
    """ Query the Together AI model with FAISS knowledge only """
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
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
    query = st.text_input("اپنا سوال درج کریں:")
    query = GoogleTranslator(source='auto', target='ur').translate(query)

    if st.button("پوچھیں"):
        if query.strip():
            with st.spinner("⏳ سوچ رہا ہے..."):
                try:
                    start_time = time.time()  # Start timing
                    
                    # Retrieve relevant documents
                    docs = retriever.get_relevant_documents(query)
                    print(docs)
                    context = "\n".join([doc.page_content for doc in docs]) if docs else "کوئی متعلقہ معلومات نہیں ملی۔"

                    # Ensure the model doesn't make up answers
                    if context == "کوئی متعلقہ معلومات نہیں ملی۔":
                        response_text = "⚠️ میں صرف قرآن سے متعلق سوالات کے جوابات دے سکتا ہوں۔"
                    else:
                        response_text = query_together_ai(context, query)

                    end_time = time.time()  # End timing
                    response_time = round(end_time - start_time, 2)  # Calculate response time

                    # Display response
                    st.subheader("**جواب:**")
                    st.write(response_text.strip())
                    st.write(f"⏱ جواب دینے کا وقت: {response_time} سیکنڈ")

                    # Display sources
                    if docs:
                        st.subheader("📌 ماخذ:")
                        for i, doc in enumerate(docs, 1):
                            source = doc.metadata.get('source', 'نامعلوم ماخذ')
                            st.write(f"{i}. {source if source.strip() else 'ماخذ دستیاب نہیں'}")
                    else:
                          st.info("کوئی متعلقہ ماخذ نہیں ملا۔")
                except Exception as e:
                    st.error(f"❌ سوال پر کارروائی کرنے میں خرابی: {e}")
        else:
            st.warning("⚠️ براہ کرم ایک درست سوال درج کریں۔")
