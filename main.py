import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import pickle
import time
import os
import tempfile

NVIDIA_API_KEY = st.secrets["API_KEY"]

#The LLM Model Defined
llm = ChatNVIDIA(
  model="tiiuae/falcon3-7b-instruct",
  temperature=0.7,
  top_p=0.7,
  max_tokens=1024,
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80,
)

#check the session memory
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None

st.title("Quick Ref - Quick refer through web and pdf")


# Initialize session state variables
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "question" not in st.session_state:
    st.session_state.question = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "sources" not in st.session_state:
    st.session_state.sources = []

#to add the URLs
st.sidebar.title("Enter the URLS")


urls = []
for x in range(3):
    url = st.sidebar.text_input(f'URL {x}')
    urls.append(url)
 

#to upload the PDFs    
st.sidebar.title("Upload the PDF")


uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")


processed = st.sidebar.button("Process")
reset = st.sidebar.button("Reset")


if reset:
    st.session_state.question = ""
    st.session_state.answer = ""
    st.session_state.sources = []


main_placeholder = st.empty()

if processed and not(uploaded_file is not None or any(url.strip() for url in urls)):
    main_placeholder.header("Without sources can not proceed...⭕⭕⭕")
    time.sleep(4)

if processed and (uploaded_file is not None or any(url.strip() for url in urls)):
    st.session_state.question = ""
    st.session_state.answer = ""
    st.session_state.sources = []
    
    splited_data = []
    
    if urls:
        main_placeholder.text('URL loading Started...✅✅✅')
        # Step 01: Loading URLs content using unstructured loader
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
                
        main_placeholder.text('Splitting Started...✅✅✅')   
        splited_url_data = splitter.split_documents(data)          
        splited_data.extend(splited_url_data)
    
    if uploaded_file:
        main_placeholder.text('PDf loading Started...✅✅✅')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        loaderPDF = PyPDFLoader(temp_file_path)
        dataPDF = loaderPDF.load()
        
        main_placeholder.text('Splitting Started...✅✅✅')   
        splited_pdf_data = splitter.split_documents(dataPDF)
        splited_data.extend(splited_pdf_data)
    
    if not splited_data:
        main_placeholder.header("Unable to load the provided sources...❌❌❌")
        time.sleep(3)
    
    if splited_data:         
        main_placeholder.text('Embedding started...✅✅✅')
        # Step 03: Embed the split data into vector form
        embedder = NVIDIAEmbeddings(model="baai/bge-m3")
        vector_db = FAISS.from_documents(splited_data, embedder)
        
        main_placeholder.text('Saving started...✅✅✅')
        # Step 04: Save the vector DB
        st.session_state.vector_db = vector_db
        main_placeholder.text('URL loaded Successfully! Now you can ask Questions ✅✅✅')
        time.sleep(2)




st.session_state.question = main_placeholder.text_input("question", st.session_state.question)



if st.session_state.question:
    if st.session_state.vector_db:
        vector_db = st.session_state.vector_db
        
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vector_db.as_retriever(),
        )
        
        result = chain({"question": st.session_state.question}, return_only_outputs=True)

        
        st.session_state.answer = result['answer']
        st.session_state.sources = result.get("sources", "").split(',')
        
        st.header("Answer : ")
        st.text(st.session_state.answer)
        
        st.header("Source : ")
        unique_sources = list(set(st.session_state.sources))
        for s in unique_sources:
            st.write(s)
    else:
        main_placeholder.header("First enter the URLs and Process!")
