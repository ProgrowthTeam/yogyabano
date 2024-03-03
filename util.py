from openai import OpenAI
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
import time
import os
from tqdm import tqdm
os.environ["OPENAI_API_KEY"] = "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"

client = OpenAI(api_key="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")
import streamlit as st
model = SentenceTransformer('all-MiniLM-L6-v2') # commnt this out after openai subscription

from pinecone import Pinecone, ServerlessSpec


pc = Pinecone(api_key='dcdc4987-797f-4e98-b014-6e652f69b207')
# openai_vectorizer = OpenAIEmbeddings() <- uncomment this 
index_name = ''

def index_init(name: str, dims: int, index_exists: bool):
    global index_name
    if not index_exists:
        pc.create_index(
            name=name,
            dimension=dims,
            spec=ServerlessSpec(cloud='aws', region='us-west-2')
        )
    index_name=name
    
def get_index():
    return pc.Index(index_name)

def insert_kb_vectors(kb_path: str, name: str, dims: int, index_exists: bool):
    index_init(name, dims, index_exists)
        
    index = get_index()
    doc_reader = PdfReader(kb_path)
    raw_text = ''
    for i, page in tqdm(enumerate(doc_reader.pages), desc='reading pdf'):
        text = page.extract_text()
        if text:
            raw_text += text
            
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200, #striding over the text
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)
    
    vector_list = []
    
    for i, text in tqdm(enumerate(texts), desc='uploading vectors'):
        # vector = {'id': str(i), "values":openai_vectorizer.embed_query(text), "metadata": {"text": text}}
        vector = {'id': str(i), "values":model.encode(text).tolist(), "metadata": {"text": text}}
        vector_list.append(vector)
        time.sleep(1)
        
    index.upsert(
        vectors=vector_list
    )

def find_match(input):
    # input_em = openai_vectorizer.embed_query(input)
    input_em = model.encode(input).tolist()
    index = get_index()
    result = index.query(vector=input_em, top_k=2, includeMetadata=True)
    print(result)
    match_text = ''
    for match in result.get('matches', []):
        match_text += match.get('metadata', {}).get('text', '') + "\n"
    return match_text

def query_refiner(conversation, query):

    response = client.completions.create(model="gpt-3.5-turbo-instruct",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0,
    max_tokens=256,)
    return response.choices[0].text
    
    # return query

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string