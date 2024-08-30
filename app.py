import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import sentence_transformers
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

st.title("Finsafe Q&A Assistant")

with st.expander("ℹ️ Disclaimer"):
    st.caption(
        """This chatbot is designed to assist users with queries related to Finsafe Organisation, providing concise and accurate responses."""
    )

# Load environment variables for API keys
groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") # huggingface embedding model
persist_directory = 'D:\Finsafe\chroma_DB'                                                           
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)  # chroma db vectordatabase

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Initialize the LLM and create the prompt
template = """You are an intelligent and concise assistant for the Finsafe organization.
Use the provided context to answer the user's question clearly and accurately in a conversational manner.
If you don't know the answer, simply state that you don't know—do not fabricate a response.
Keep your answers to no more than three sentences and always conclude with, 'Thanks for asking!'
Additionally, include this statement: 'For more inquiries, contact +91 7411677575 or reach out via email at support@finsafe.in.'

{context}

Question: {question}

Helpful Answer:"""

prompt = PromptTemplate(input_variables=["question"], template=template)

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")   # Initialize the LLMChain with ChatGroq
                        
# Chat History Management
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input and chatbot response handling
if user_input := st.chat_input("What's your question?"):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Format documents for context
    def format_docs(pages):
        return "\n\n".join(doc.page_content for doc in pages)

    # Create a RAG chain for dynamic retrieval and response
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Add a spinner while processing the response
    with st.spinner('Thinking...'):
        response = rag_chain.invoke(user_input)
        
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.chat_history.append({'human': user_input, 'AI': response})
