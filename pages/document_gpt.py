import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma
from langchain.storage import LocalFileStore
from langchain.prompts import chatprompttemplate
from langchain.callbacks.base import basecallbackhandler

class chatcallbackhandler(basecallbackhandler):
    
    messasge=""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token:str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message) 

llm = ChatOpenAI(
    temperature = 0.1,
    streaming=True,
    callbacks=[chatcallbackhandler(),]
    )

if "messages" not in st.session_state:
    st.session_state["messages"]=[]

@st.cache_data

def embed_file(file):
    st.write(file)
    file_content=file.read()
    file_path=f"./.cache/files/{file.name}"
    with open(file_path,"wb") as f:
        f.write(file_content)
    cache_dir=LocalFileStore(f"./.cache/embeddings/{file.name}")
    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200,  
        chunk_overlap=50,
        separator="\n",
    )
    docs = loader.load_and_split(text_splitter=splitter)
    embedding = OpenAIEmbeddings()
    cached_embeddings=CacheBackedEmbeddings.from_bytes_store(
        embedding, cache_dir
    )
    vectorstore = Chroma.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message():
    st.session_state["messages"].append({"message":message, "role":role})

def send_message(message,role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_documents(docs):
    return docs = "\n\n".join(document.page_content for document in docs)


prompt = chatprompttemplate.from_messages([
    ("system",
     """Answer the question using ONLY the following context. If you don't know the answer just say you don't know. 
     DON'T make anything up.
     
     Context:{context}""")
    ("human","{question}")
])

st.set_page_config(
    page_title="Document GPT",
)
st.title("Document GPT")

st.markdown("""
            Welcome! 
            
            Use this chatbot to ask questions about your files to your AI!
            """)

with st.sidebar:
    file = st.file_uploader("Upload a .txt, .pdf, or .docx file", 
                            type=["pdf", "txt", "docx"])

if file:
    retriever = embed_file(file)
    send_message("I'm ready!","ai")
    paint_history()
    message = st.chat_input("Ask anything about your file..")
    if message:
        send_message(message, "human")
        chain = {
            "context": retriever | Runnablelambda(format_documents),
            "question": Runnablepassthrough()
        }| prompt | llm \
    with st.chat_message("ai"): 
        response = chain.invoke(message)
   

else:
    st.session_state["messages"] = []
