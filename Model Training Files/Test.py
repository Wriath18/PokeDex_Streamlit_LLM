# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import os
# load_dotenv()

# ## load the GROQ And OpenAI API KEY 
# groq_api_key=os.getenv('GROQ_API_KEY')
# os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

# st.title("Gemma Model Document Q&A")

# llm=ChatGroq(groq_api_key=groq_api_key,
#              model_name="Llama3-8b-8192")

# prompt=ChatPromptTemplate.from_template(
# """
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question
# <context>
# {context}
# <context>
# Questions:{input}

# """
# )

# def vector_embedding():

#     if "vectors" not in st.session_state:

#         st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
#         st.session_state.loader=PyPDFDirectoryLoader("./") ## Data Ingestion
#         st.session_state.docs=st.session_state.loader.load() ## Document Loading
#         st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
#         st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
#         st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings





# prompt1=st.text_input("Enter Your Question From Doduments")


# if st.button("Documents Embedding"):
#     vector_embedding()
#     st.write("Vector Store DB Is Ready")

# import time



# if prompt1:
#     document_chain=create_stuff_documents_chain(llm,prompt)
#     retriever=st.session_state.vectors.as_retriever()
#     retrieval_chain=create_retrieval_chain(retriever,document_chain)
#     start=time.process_time()
#     response=retrieval_chain.invoke({'input':prompt1})
#     print("Response time :",time.process_time()-start)
#     st.write(response['answer'])

#     # With a streamlit expander
#     with st.expander("Document Similarity Search"):
#         # Find the relevant chunks
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------")




# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import time

# load_dotenv()

# ## Load the GROQ and OpenAI API key
# groq_api_key = os.getenv('GROQ_API_KEY')
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# st.title("Gemma Model Document Q&A")

# llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# prompt = ChatPromptTemplate.from_template(
#     """
#     Answer the questions based on the provided context only.
#     Please provide the most accurate response based on the question.
#     <context>
#     {context}
#     <context>
#     Questions: {input}
#     """
# )

# def vector_embedding():
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         st.session_state.loader = PyPDFDirectoryLoader("./")  # Data ingestion
#         st.session_state.docs = st.session_state.loader.load()  # Document loading
#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk creation
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings

# prompt1 = st.text_input("Enter Your Question From Documents")

# if st.button("Documents Embedding"):
#     vector_embedding()
#     st.write("Vector Store DB is ready")

# if prompt1:
#     document_chain = create_stuff_documents_chain(llm, prompt)
#     if "vectors" not in st.session_state:
#         vector_embedding()
#     retriever = st.session_state.vectors.as_retriever()
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)
#     start = time.process_time()
#     response = retrieval_chain.invoke({'input': prompt1})
#     st.write(f"Response time: {time.process_time() - start} seconds")
#     st.write(response['answer'])

#     # With a Streamlit expander
#     with st.expander("Document Similarity Search"):
#         # Find the relevant chunks
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------")

# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import time
# import pokemon_detector  # Ensure you have this module available
# from transformers import pipeline
# import fitz  # PyMuPDF

# load_dotenv()

# ## Load the GROQ and OpenAI API key
# groq_api_key = os.getenv('GROQ_API_KEY')
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# def vector_embedding():
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         st.session_state.loader = PyPDFDirectoryLoader("./")  # Data ingestion
#         st.session_state.docs = st.session_state.loader.load()  # Document loading
#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk creation
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings

# def main():
#     st.title("PokeDex using CNN (MobileNetV2)")
#     st.text("Upload your image")

#     if "classification_done" not in st.session_state:
#         st.session_state.classification_done = False

#     upload_file = st.file_uploader("Choose an image ... ", type=['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'])

#     if upload_file is not None:
#         st.image(upload_file, caption='Uploaded Image', use_column_width=True)

#         if st.button('Classify') and not st.session_state.classification_done:
#             result = pokemon_detector.predict(upload_file)
#             st.success(f"The pokemon has been classified as {result}")
#             st.session_state.classification_done = True

#     if st.session_state.classification_done:
#         st.title("Gemma Model Document Q&A")

#         if st.button("Documents Embedding"):
#             vector_embedding()
#             st.write("Vector Store DB is ready")
#             st.session_state.embedding_done = True

#         prompt1 = st.text_input("Enter Your Question From Documents")

#         if prompt1:
#             document_chain = create_stuff_documents_chain(llm, prompt)
#             if "vectors" not in st.session_state:
#                 vector_embedding()
#             retriever = st.session_state.vectors.as_retriever()
#             retrieval_chain = create_retrieval_chain(retriever, document_chain)
#             start = time.process_time()
#             response = retrieval_chain.invoke({'input': prompt1})
#             st.write(f"Response time: {time.process_time() - start} seconds")
#             st.write(response['answer'])

#             # With a Streamlit expander
#             with st.expander("Document Similarity Search"):
#                 # Find the relevant chunks
#                 for i, doc in enumerate(response["context"]):
#                     st.write(doc.page_content)
#                     st.write("--------------------------------")

# if __name__ == '__main__':
#     load_dotenv()
#     groq_api_key = os.getenv('GROQ_API_KEY')
#     os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    
#     llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
#     prompt = ChatPromptTemplate.from_template(
#         """
#         Answer the questions based on the provided context only.
#         Please provide the most accurate response based on the question.
#         <context>
#         {context}
#         <context>
#         Questions: {input}
#         """
#     )

#     main()


import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
import pokemon_detector  # Ensure you have this module available
from transformers import pipeline
import fitz  # PyMuPDF

load_dotenv()

## Load the GROQ and OpenAI API key
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./")  # Data ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings

def main():
    st.title("PokeDex using CNN (MobileNetV2)")
    st.text("Upload your image")

    if "classification_done" not in st.session_state:
        st.session_state.classification_done = False

    if "embedding_done" not in st.session_state:
        st.session_state.embedding_done = False

    if "description_generated" not in st.session_state:
        st.session_state.description_generated = False

    upload_file = st.file_uploader("Choose an image ... ", type=['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'])

    if upload_file is not None:
        st.image(upload_file, caption='Uploaded Image', use_column_width=True)

        if st.button('Classify') and not st.session_state.classification_done:
            result = pokemon_detector.predict(upload_file)
            st.success(f"The pokemon has been classified as {result}")
            st.session_state.classification_done = True

            # Automatically perform document embedding
            vector_embedding()
            st.write("Vector Store DB is ready")
            st.session_state.embedding_done = True

            # Automatically generate a description for the classified Pokémon
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start = time.process_time()
            response = retrieval_chain.invoke({'input': f"Describe {result}"})
            st.session_state.description = response['answer']
            st.session_state.description_generated = True

    if st.session_state.classification_done and st.session_state.embedding_done:
        st.title("Gemma Powered Pokedex RAG Q&A")

        if st.session_state.description_generated:
            st.write("Description of the classified Pokémon:")
            st.write(st.session_state.description)

        prompt1 = st.text_input("Enter Your Question From Documents")

        if prompt1:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            st.write(f"Response time: {time.process_time() - start} seconds")
            st.write(response['answer'])

            # With a Streamlit expander
            with st.expander("Document Similarity Search"):
                # Find the relevant chunks
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")

if __name__ == '__main__':
    load_dotenv()
    groq_api_key = os.getenv('GROQ_API_KEY')
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )

    main()

