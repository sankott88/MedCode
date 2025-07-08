# MedAssist AI – Medical Coding Assistant (RAG-Powered)
 
**MedAssist AI** is a Retrieval-Augmented Generation (RAG) application that helps medical professionals and coders retrieve accurate medical codes (ICD-10, CPT, HCPCS) using natural language queries.
Built using LangChain, OpenAI, Pinecone, and Streamlit.
 
---
 
## Features
 
- Ask clinical or billing-related questions in natural language
- Gets context from real-world medical documents (PDFs, XMLs, etc.)
- Returns answers in a Markdown table format with:
  - **Code Type**
  - **Code**
  - **Description**
 
---
 
## Technologies Used
 
- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [OpenAI GPT (text-embedding-3-small)](https://platform.openai.com/docs/)
- [Pinecone](https://www.pinecone.io/) for vector storage
- [Unstructured](https://github.com/Unstructured-IO/unstructured) for document parsing
 
---
 
## How to Run Locally
 
1. Clone this repository:
   git clone https://github.com/yourname/medassist-app.git
   cd medassist-app
 
2. Install dependencies:
pip install -r requirements.txt
 
3. Create a .env file in the root directory and add your API keys:
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=your-pinecone-key
 
4. Run the app:
streamlit run app.py
