import os
import streamlit as st
import pickle
import time
from langchain_google_genai import ChatGoogleGenerativeAI  # Changed
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Updated path
from langchain_community.document_loaders import UnstructuredURLLoader  # Updated path
from langchain_huggingface import HuggingFaceEmbeddings  # Changed to HuggingFace
from langchain_community.vectorstores import FAISS  # Updated path

from dotenv import load_dotenv

load_dotenv() # take environment variables from .env

# Building the UI wirth Streamlit
st.title("News Research Tool")
st.sidebar.title("News Article URLs")

urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "notebooks/faise_store_geminiai.pkl"

main_placeholder = st.empty()

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.9,
    max_output_tokens=500
)

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading... Started...")
    data = loader.load()
    # split the data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter... Started...")
    docs = text_splitter.split_documents(data)

    # create embeddings and save it to the FAISS index
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorindex = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embeddings Vector... Started Building...")
    time.sleep(2)
    # save the FAISS index to the pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorindex, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorindex = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorindex.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # {"answer": "", "sources": []}
            st.header("Answer")
            st.subheader(result["answer"])
