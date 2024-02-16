#!/usr/bin/env python
# coding: utf-8

# In[33]:


get_ipython().system('pip install langchain')
get_ipython().system('pip install openai')
get_ipython().system('pip install PyPDF2')
get_ipython().system('pip install faiss-cpu')
get_ipython().system('pip install tiktoken')


# In[17]:


pip install pinecone-client


# In[35]:


get_ipython().system('pip install langchain --upgrade')

get_ipython().system('pip install pypdf')


# In[36]:


from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()


# In[37]:


loader = TextLoader(file_path="C:\\Users\\wwwmr\Downloads\HealthSanitaryInspectorTT.txt")


# In[38]:


loader


# In[39]:


data = loader.load()


# In[15]:


# Here's an example of the first document that was returned
for doc in docs:
    print (f"{doc.page_content}\n")


# In[13]:


# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# texts = text_splitter.split_documents(data)


# In[14]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(data)


# In[17]:


from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone


# In[16]:


PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', 'c4cee698-858d-4eac-b27d-c9d5db4728f4')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV', 'us-east1-gcp') # You may need to switch with your env

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "langchaintest" # put in the name of your pinecone index here

docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)


# In[29]:


from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone


# In[31]:


pip install langchain


# In[ ]:


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKey')


# In[ ]:


# Note: If you're using PyPDFLoader then it will split by page for you already
print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[0].page_content)} characters in your sample document')
print (f'Here is a sample: {data[0].page_content[:200]}')


# In[32]:


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


# In[40]:


# load it into Chroma
vectorstore = Chroma.from_documents(texts, embeddings)


# In[ ]:


from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain


# In[ ]:


query = "What is great about having kids?"
docs = vectorstore.similarity_search(query)

