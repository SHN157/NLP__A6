# A6: Let's Talk with Yourself

## Table of Contents
- [Student Information](#student-information)
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [System Components](#system-components)
- [Implementation Details](#implementation-details)
- [Analysis of Issues Related to Unrelated Information](#analysis-of-issues-related-to-unrelated-information)
- [Mitigation Strategies](#mitigation-strategies)
- [Running the Web Application](#running-the-web-application)
- [File Structure](#file-structure)
- [Deliverables](#deliverables)
- [Conclusion](#conclusion)

## Student Information
- **Name**: Soe Htet Naing  
- **ID**: st125166  

## Project Overview
This project applies **Retrieval-Augmented Generation (RAG)** techniques using the **LangChain framework** to build a chatbot that specializes in answering questions related to personal information, documents, and relevant sources. The chatbot retrieves contextual information from a FAISS vector database and generates responses using a **Groq-powered Llama-3.3-70b-specdec model** and **FastChat-T5-3B-v1.0**.

## Technologies Used
- **LangChain**: For RAG pipeline implementation.
- **FAISS**: For storing and retrieving document embeddings.
- **HuggingFace Instruct Embeddings (hkunlp/instructor-base)**: For embedding personal documents.
- **Groq’s Llama-3.3-70b-specdec**: For generating responses.
- **FastChat-T5-3B-v1.0**: For additional response generation.
- **Streamlit**: For building a web application interface.
- **Transformers**: For loading and running the FastChat model.

## System Components
### 1) Retriever Model
- **Model Used**: FAISS with `HuggingFaceInstructEmbeddings (hkunlp/instructor-base)`
- **Purpose**: Retrieves the most relevant document chunks based on the user's query.

### 2) Generator Models
- **Model Used**: Groq’s `llama-3.3-70b-specdec`
- **Model Used**: FastChat-T5-3B-v1.0 ([Hugging Face Link](https://huggingface.co/lmsys/fastchat-t5-3b-v1.0))
- **Purpose**: Generates human-like responses based on retrieved document context.

## Implementation Details
### 1) Document Processing
- **PDF Loading**: `PyPDFLoader` extracts text from personal documents.
- **Text Splitting**: `RecursiveCharacterTextSplitter` breaks text into chunks of **70 characters** with **10-character overlap**.
- **Embedding Generation**: Documents are converted into vector embeddings using `HuggingFaceInstructEmbeddings`.
- **FAISS Indexing**: The vector representations are stored in a FAISS database for retrieval.

### 2) Chatbot Response Generation
- User inputs a question.
- The retriever fetches relevant document chunks.
- **Llama-3.3-70b-specdec** and **FastChat-T5-3B-v1.0** generate responses based on retrieved content.
- The chatbot displays the response along with supporting document references.

## Analysis of Issues Related to Unrelated Information
### **Retriever Model Issues (FAISS with HuggingFaceInstructEmbeddings)**
- **Semantic Mismatch**: FAISS may retrieve documents based on word similarity rather than true contextual relevance.
- **Context Fragmentation**: Small document chunks may lack full context, leading to incomplete answers.
- **Embedding Limitations**: The embedding model may not fully capture nuanced queries, reducing retrieval quality.

### **Generator Model Issues (Llama-3.3-70b-specdec & FastChat-T5-3B-v1.0)**
- **Hallucination**: The models may generate responses that are plausible but not based on retrieved documents.
- **Over-Reliance on Prompting**: If retrieval is weak, the models may fall back on pre-trained knowledge instead of document-based responses.
- **Bias in Generation**: The models may introduce biases that are not present in the original retrieved documents.

## Mitigation Strategies
### **Enhancing Retrieval Accuracy**
- Implement **hybrid search** (dense embeddings + keyword search) for improved context relevance.
- Increase **text chunk size and overlap** to retain more context.

### **Improving Prompt Engineering**
- Modify prompts to **prioritize retrieved documents** over the model’s pre-trained knowledge.

### **Post-Processing Filters**
- Apply **confidence scoring** to flag and filter hallucinated responses.
- Use **reinforcement learning-based fine-tuning** to reduce off-topic generations.

## Running the Web Application
### Installation
```bash
pip install streamlit langchain faiss-cpu torch transformers langchain-groq
```
### Running the App
```bash
streamlit run chatbot_app.py
```

## File Structure
The project follows this directory structure:
```
NLP_A6/
│── app/                    # Web application folder
│   │── vector/soe/         # FAISS vector database storage
│   │   │── shn.faiss       # FAISS index file
│   │   │── shn.pkl         # FAISS metadata file
│   │── app.py              # Main web application script
│
│── embedding_model/        # Pretrained embedding model storage
│   │── instructor_model.pth
│
│── chatbot_responses.json  # Chatbot-generated Q&A responses
│── A6.ipynb                # Main Jupyter Notebook for chatbot implementation
│── fastchat_t5_3b.ipynb    # Notebook for FastChat-T5 model
│── Soe_Htet_Naing_Resume.pdf  # Resume document used for chatbot retrieval
```
### **Explanation**
- `app/` contains the Streamlit web application.
- `vector/soe/` holds the FAISS database storing document embeddings.
- `embedding_model/` contains the pretrained instructor embedding model.
- `chatbot_responses.json` stores chatbot-generated question-answer pairs.
- `A6.ipynb` is the primary notebook for implementing the chatbot.
- `fastchat_t5_3b.ipynb` handles the FastChat-T5 model setup.
- `Soe_Htet_Naing_Resume.pdf` is used as source material for retrieval.

## Deliverables
- **GitHub Repository** with:
  - Jupyter notebook containing code implementation.
  - `README.md` with project documentation.
  - Web application folder named `app/`.
- **Question-Answer JSON File** containing chatbot-generated responses:
  ```json
  [
    { "question": "How old are you?", "answer": "To determine Soe Htet Naing's age, we need to consider the current year and their birth year. Since the cutting knowledge date is December 2023, let's use that as a reference point. Soe Htet Naing was born in 1997. If we calculate their age as of 2023: 2023 - 1997 = 26 So, as of 2023, Soe Htet Naing is 26 years old. Please note that this calculation is based on the given knowledge cutoff date and may not reflect their current age if the current year is later than 2023." },
    { "question": "What is your highest level of education?", "answer": "Soe Htet Naing is currently a Master's student. His highest completed degree is a Bachelor's." }
  ]
  ```

## Conclusion
This project successfully implements a **RAG-based chatbot** capable of retrieving and generating accurate responses based on personal documents. The integration of **FAISS for retrieval**, **Llama-3.3-70b-specdec**, and **FastChat-T5-3B-v1.0** enhances response accuracy. Future improvements include refining **retrieval accuracy**, **prompt optimization**, and **bias mitigation techniques**.
