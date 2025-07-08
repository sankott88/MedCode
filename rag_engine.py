import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import Pinecone as LangChainPinecone
from langchain_community.vectorstores import Pinecone as VectorStorePinecone
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_type=os.environ["OPENAI_API_KEY"]
)

#Setup LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

prompt_template = PromptTemplate.from_template("""
You are a helpful medical coding assistant.
User the following context (medical documents and codebooks) to answer the question.
Provide your answer in a ""Markdown table""  with 3 columns:
**Code Type**,**Code**, and **Description**.
If no codes are found, respond with "No relevant codes found".
Context: {context}
Question: {question}
""")

#Load vector store
vector_store = VectorStorePinecone.from_existing_index(
    index_name="medassist-index",
    embedding=embedding_model
)

# Setup retriever chain
retriever = vector_store.as_retriever() #LLM "look up" relevant chunks from index
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever = retriever,
    chain_type ="stuff",
    chain_type_kwargs={"prompt":prompt_template}
)

# This function takes a question, sends it to the retriever + GPT, and returns the final answer
def ask_medical_question(user_question: str) -> str:
    response = qa_chain.invoke(user_question) # send question to the RetrievalQA chain
    answer = response["result"] # Extract the answer from the response 
    return answer # Return the answer