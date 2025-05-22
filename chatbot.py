import os
from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

app = Flask(__name__)

groq_api_key = os.getenv("GROQ_API_KEY")

def load_and_index_pdf(pdf_path="company_data.pdf"):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embedding)
    db.save_local("company_faiss_index")

def setup_qa():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if not os.path.exists("company_faiss_index"):
        load_and_index_pdf()

    db = FAISS.load_local("company_faiss_index", embedding, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatGroq(model_name="llama3-70b-8192", api_key=groq_api_key)

    prompt = PromptTemplate.from_template("""
You are a helpful assistant for a digital marketing company.
Try to answer the user's question based on the provided context from the company document.
If the answer is not found in the context, provide a helpful and accurate answer from your own knowledge, focusing on digital marketing topics.

Context:
{context}

Question:
{question}
""")

    qa_chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=load_qa_chain(llm, chain_type="stuff", prompt=prompt),
        return_source_documents=False
    )
    return qa_chain

qa_chain = setup_qa()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query", "")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    response = qa_chain.run(user_query)
    return jsonify({"response": response})


app.run(host="0.0.0.0", port=8000)
