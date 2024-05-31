from openai import OpenAI
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from IPython.display import display, Image, Audio

import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
import os
import requests
from tqdm import tqdm

client = OpenAI()
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

OK = "OK"


pc = Pinecone()
# openai_vectorizer = OpenAIEmbeddings() <- uncomment this 
index_name = ''
embeddings = OpenAIEmbeddings()

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


def find_match(input):
    # input_em = openai_vectorizer.embed_query(input)
    input_em = embeddings.embed_query(input) 
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

def upload_file_to_pinecone(raw_text, file_name):
    # Splitting up the text into smaller chunks for indexing
    try:
        text_splitter = CharacterTextSplitter(        
            separator = "\n",
            chunk_size = 10000,
            chunk_overlap  = 200, #striding over the text
            length_function = len,
        )
        
        texts = text_splitter.split_text(raw_text)
        index = get_index()
        embeddings = OpenAIEmbeddings()
        batch_size = max(len(texts) // 10, 1)

        for i in range(0, len(texts), batch_size):
            embeds = []
            batch = texts[i:i+batch_size]
            vectors = embeddings.embed_documents(batch)
            for j, vector in enumerate(vectors):
                embed = {'id': f'{i}_{j}', "values": vector, "metadata": {"file_name": file_name, "text": batch[j]}}
                embeds.append(embed)
            
            index.upsert(
                vectors=embeds
            )
        
        return OK
        
    except Exception as e:
        return e

def video_to_text(path):

    video = cv2.VideoCapture(path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    frames = len(base64Frames)
    BATCH_SIZE=1000
    text = ""
    
    for i in range(0, frames, BATCH_SIZE):
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [
                    "These are frames from a video uploaded to a company's knowledge base. Generate an elaborate description of what's happening in the video, which can be used to ask further questions by users about the video.",
                    *map(lambda x: {"image": x, "resize": 768}, base64Frames[i:i+BATCH_SIZE:50]),
                ],
            },
        ]
        params = {
            "model": "gpt-4o",
            "messages": PROMPT_MESSAGES,
            "max_tokens": 2048,
        }
        
        result = client.chat.completions.create(**params)
        text += result.choices[0].message.content
    return frames, text

def audio_to_text(path):
    audio_file = open(path, "rb")
    translation = client.audio.translations.create(
        model="whisper-1", 
        file=audio_file
    )
    return translation.text
