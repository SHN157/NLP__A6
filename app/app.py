import streamlit as st
import os
import torch
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceInstructEmbeddings




# # Set Hugging Face API token
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your key" 

os.environ["GROQ_API_KEY"] = "your key"

def load_faiss():
    vector_path = "./vector"
    db_file_name = "soe"
    embedding_model = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-base"
    )
    vectordb = FAISS.load_local(
        folder_path=os.path.join(vector_path, db_file_name),
        embeddings=embedding_model,
        index_name="shn",
        allow_dangerous_deserialization=True
    )
    return vectordb.as_retriever()

retriever = load_faiss()
groq_model = ChatGroq(model_name="llama-3.3-70b-specdec", temperature=0.7)

prompt_template = """
Please answer the following question accurately based on the provided context of a person named Soe Htet Naing.
Context:
{context}

Question: {question}

Gentle & Informative Answer:
""".strip()

PROMPT = PromptTemplate.from_template(prompt_template)

def get_response(question):
    input_document = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in input_document])
    response = groq_model.invoke(PROMPT.format(context=context, question=question))
    return response, input_document

# Streamlit UI Setup
st.set_page_config(page_title="Chatbot with RAG", layout="wide")
st.title("Chatbot - Document-Supported Q&A")
st.write("Type your question below and get an AI-generated response with supporting sources.")

user_input = st.text_input("Ask a question:", "")

if user_input:
    response, documents = get_response(user_input)
    
    # Display chatbot response
    st.subheader("AI Response")
    st.write(response.content)
    
    # Display supporting documents
    st.subheader("Supporting Documents")
    for doc in documents:
        st.markdown(f"**Page Content:** {doc.page_content[:300]}...")


# # Load Groq Model (Llama3-70b)st
# groq_model = ChatGroq(model_name="llama-3.3-70b-specdec", temperature=0.7)

# # ✅ Load the SAME embedding model used when training FAISS
# # embedding_model = HuggingFaceInstructEmbeddings(
# #     model_name="hkunlp/instructor-base"
# # )ss

# # embedding_model= SentenceTransformerEmbeddings(model_name="hkunlp/instructor-base")

# embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Load FAISS Vector Store (Ensure the database is correctly stored)
# vector_path = "/Users/soehtetnaing/Documents/GitHub/NLP_A6/app/vector"
# db_file_name = "soe"

# retriever = FAISS.load_local(
#     folder_path=os.path.join(vector_path, db_file_name),
#     embeddings=embedding_model,  # FAISS loads embeddings internally
#     index_name="shn",
#     allow_dangerous_deserialization=True
# ).as_retriever()

# # Define Chatbot Prompt
# prompt_template = """
# Please answer the following question accurately based on the provided context of a person named Soe Htet Naing.

# Context:
# {context}

# Question: {question}

# Gentle & Informative Answer:
# """.strip()

# PROMPT = PromptTemplate.from_template(prompt_template)

# # Streamlit UI Setup
# st.set_page_config(page_title="Soe Htet Naing Chatbot", layout="wide")
# st.title("🤖 Soe Htet Naing Chatbot")

# # Initialize Chat History
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # User Input
# user_input = st.text_input("Type your message here:", key="input")

# if st.button("Send"):
#     if user_input:
#         # Retrieve relevant documents
#         retrieved_docs = retriever.get_relevant_documents(user_input)
#         context = "\n".join([doc.page_content[:500] for doc in retrieved_docs])  # First 500 chars

#         # Format prompt with retrieved context
#         formatted_prompt = PROMPT.format(context=context, question=user_input)

#         # Generate response
#         response = groq_model.invoke(formatted_prompt)

#         # Store conversation history
#         st.session_state.chat_history.append(("You", user_input))
#         st.session_state.chat_history.append(("Chatbot", response))

#         # Display chat history
#         for speaker, text in st.session_state.chat_history:
#             st.write(f"**{speaker}:** {text}")

#         # Show retrieved documents
#         st.subheader("📄 Supporting Documents:")
#         for i, doc in enumerate(retrieved_docs):
#             st.write(f"🔹 **Document {i+1}:** {doc.page_content[:300]}...")  # Display snippet

