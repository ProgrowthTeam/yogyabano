from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from util import *
from google.cloud import translate_v2 as translate
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO


#change the background color of the sidebar
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#FFFFFF, #FFFFFF);
    }
    </style>
    """,
    unsafe_allow_html=True
)
#change colour of the text in the sidebar
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        color: #2F4362;
    }
    </style>
    """,
    unsafe_allow_html=True
)


#change background color of app
st.markdown(
    """
    <style>
    .reportview-container {
        background: #FFFFFF
    }
    </style>
    """,
    unsafe_allow_html=True
)

#change the background color of the main content
st.markdown(
    """
    <style>
    .main {
        background: #FF7500
    }
    </style>
    """,
    unsafe_allow_html=True
)


translate_client = translate.Client()

def translate_text(target: str, text: str) -> dict:
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)

    print("Text: {}".format(result["input"]))
    print("Translation: {}".format(result["translatedText"]))
    print("Detected source language: {}".format(result["detectedSourceLanguage"]))

    return result


# insert_kb_vectors("/home/kamal/Downloads/paintermanual.pdf", 'sentence-transform-embed-chatbot', 384, false)
index_init('fitter', 1536, True)
st.title("Yogyabano                  Empowering Skill Training with AI")
st.subheader("Saarthi - Your Personalized AI Trainer for Skill Training,")

#add a dropdown menu in streamlit 
st.sidebar.title("Choose a Language")
language = st.sidebar.selectbox(
    'Select a language',
    ['English', 'Hindi', 'Tamil', 'Telugu', 'Kannada', 'Malayalam', 'Bengali', 'Gujarati', 'Marathi', 'Punjabi', 'Odia', 'Assamese', 'Urdu', 'Sanskrit']
)




iso_codes = {
    'English': 'en',
    'Hindi': 'hi',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Kannada': 'kn',
    'Malayalam': 'ml',
    'Bengali': 'bn',
    'Gujarati': 'gu',
    'Marathi': 'mr',
    'Punjabi': 'pa',
    'Odia': 'or',
    'Assamese': 'as',
    'Urdu': 'ur',
    'Sanskrit': 'sa'
}
#remember the language that was choosen
st.session_state['language'] = language

import streamlit as st 

st.sidebar.subheader("login")
username = st.sidebar.text_input("Username")
phone = st.sidebar.text_input("phone number")
Field = st.sidebar.text_input("Field of interest")

button_was_clicked = st.sidebar.button("SUBMIT")
if button_was_clicked:
    st.sidebar.write("Username:", username)
    st.sidebar.write("Phone:", phone)
    st.sidebar.write("Field:", Field)

@st.cache_data
def convert_pdf_to_txt_file(path):
    texts = []
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    # fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    
    file_pages = PDFPage.get_pages(path)
    nbPages = len(list(file_pages))
    for page in PDFPage.get_pages(path):
      interpreter.process_page(page)
      t = retstr.getvalue()
    # text = retstr.getvalue()

    # fp.close()
    device.close()
    retstr.close()
    return t

def main():
   
    
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # To read file as bytes:
        
        name = uploaded_file.name
        st.write("File uploaded successfully!")
                
        raw_text = convert_pdf_to_txt_file(uploaded_file)
        
        st.write("Uploading file to vector db...")
        
        status = upload_file_to_pinecone(raw_text, name)
        
        if status == OK:
            st.write("File embedded to index!")
        else:
            st.write(f"Error: {status}")
        
 
if __name__ == "__main__":
    main()






if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can i assist you"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)

if 'buffer_memory' not in st.session_state:
    st.session_state['buffer_memory']=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'currently we dont know this but you can contact us at yogyabano.com'""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state['buffer_memory'], prompt=prompt_template, llm=llm, verbose=True)




# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            translated_text = translate_text('en', query)["translatedText"]
            # st.code(conversation_string)
            refined_query = query_refiner(translated_text, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)
            # print(context)  
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            translated_response = translate_text(iso_codes[st.session_state['language']], response)["translatedText"]
        st.session_state.requests.append(query)
        st.session_state.responses.append(translated_response) 
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

          
