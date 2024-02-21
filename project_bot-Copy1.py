#!/usr/bin/env python
# coding: utf-8

# In[58]:


get_ipython().system('pip -q install langchain openai tiktoken PyPDF2 faiss-cpu')


# In[59]:


get_ipython().system('pip -q install langchain-pinecone')


# In[60]:


get_ipython().system('pip install dotenv')


# In[65]:


import os

os.environ["OPENAI_API_KEY"] = "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"


# In[66]:


get_ipython().system('pip show langchain')


# In[67]:


from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS 


# In[68]:


doc_reader = PdfReader("C:/Users/wwwmr/Downloads/PainterGeneral1stYearVolI.pdf")


# In[69]:


doc_reader


# In[70]:


print(doc_reader)


# In[71]:


# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(doc_reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text


# In[72]:


len(raw_text)


# In[73]:


# Splitting up the text into smaller chunks for indexing
text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200, #striding over the text
    length_function = len,
)
texts = text_splitter.split_text(raw_text)


# In[74]:


len(texts)


# In[75]:


# Splitting up the text into smaller chunks for indexing
text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200, #striding over the text
    length_function = len,
)
texts = text_splitter.split_text(raw_text)


# In[76]:


texts[20]


# In[77]:


embeddings = OpenAIEmbeddings()


# In[78]:


docsearch = FAISS.from_texts(texts, embeddings)


# In[79]:


docsearch.embedding_function


# In[80]:


query = "what is painting"
docs = docsearch.similarity_search(query)


# In[81]:


len(docs)


# In[82]:


docs[0]


# In[83]:


from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


# In[84]:


chain = load_qa_chain(OpenAI(), 
                      chain_type="stuff")


# In[85]:


chain.llm_chain.prompt.template


# In[86]:


query = "who are the authors of the book?"
docs = docsearch.similarity_search(query)
chain.run(input_documents=docs, question=query)


# In[87]:


query = "what is book about?"
docs = docsearch.similarity_search(query)
chain.run(input_documents=docs, question=query)


# In[88]:


get_ipython().system('pip install pinecone-client')
from langchain_pinecone import Pinecone



# In[91]:


import os
PINECONE_API_KEY='c4cee698-858d-4eac-b27d-c9d5db4728f4'
PINECONE_API_ENV='gcp-starter'

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['PINECONE_API_ENV'] = PINECONE_API_ENV

# # initialize pinecone
# PC(
#     api_key=PINECONE_API_KEY,  
#     environment=PINECONE_API_ENV  # next to api key in console
# )
index_name = "chatbot2" # put in the name of your pinecone index here

docsearch = Pinecone.from_texts(texts, embeddings, index_name=index_name)


# In[92]:


query = "What is this book?"
docs = docsearch.similarity_search(query)

chain.run(input_documents=docs, question=query)


# In[ ]:




