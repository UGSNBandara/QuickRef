from dotenv import load_dotenv
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import pickle
import time
import os


load_dotenv()

llm = ChatNVIDIA(
  model="tiiuae/falcon3-7b-instruct",
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
)

st.title("Document Based Answering")

st.sidebar.title("Enter yout URLS")

urls = []
for x in range(3):
    url = st.sidebar.text_input(f'URL {x}')
    urls.append(url)
    
processed = st.sidebar.button("Process")


main_placeholder = st.empty()
file_path = 'VDB_store/vdb.pkl'

if processed:
    
    main_placeholder.text('URL are loading...')
    #step 01 loading urls content using unstructured loader
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    
    main_placeholder.text('Loaded Data splitting into chunks...')    
    #step 02 Split the loaded data into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
    )
    
    splited_data = splitter.split_documents(data)
    
    main_placeholder.text('Embading...')
    #step 03 Embade the splited data into vector form
    embedder = NVIDIAEmbeddings(model="baai/bge-m3")
    vector_db = FAISS.from_documents(splited_data, embedder)
    
    main_placeholder.text('Saving...')
    #step 04 save the vector db
    
    with open(file_path, 'wb') as f:
        pickle.dump(vector_db, f)
    main_placeholder.text('vectore db has Saved Succussfuly ! Now you can ask Questions')
    time.sleep(2)
    
question = main_placeholder.text_input("question")

if question:
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            vector_db = pickle.load(f)
        
        chain = RetrievalQAWithSourcesChain.from_llm(
                llm = llm,
                retriever = vector_db.as_retriever(),
                )
        
        result = chain({"question" : question}, return_only_outputs=True)
        
        st.header("Answer : ")
        
        st.text(result['answer'])
        
        st.header("Source : ")
        
        sources = result.get("sources", "")
        
        sources = sources.split(',')
        unique_sources = list(set(sources))
        for s in unique_sources:
            st.write(s)
    else:
        main_placeholder.header("First enter the URLS and Process !")    