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
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av


#change the colour theme to orange and white
#change the background color of the sidebar

#change colour of the text in the sidebar 
#change colour of the content in the sidebar
#change background color of app
st.markdown(
    """
    <style>
    .reportview-container {
        background: #EEEEEE
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
        background: #ffffff;
        
        
    }
    h1 ,p,span{
        color: #ff7500;
        
    }
    input {
        background: #ffead9;
        color: #ff7500
    }
    
    section{
        background: red;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        color: #4F6D7A;
        background: #FFFFFF
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

st.markdown(
    """
    <style>
    .title  {
        color: #4F6D7A;
        background: #FFFFFF
    }
    </style>
    """,
    unsafe_allow_html=True
)
# insert_kb_vectors("/home/kamal/Downloads/paintermanual.pdf", 'sentence-transform-embed-chatbot', 384, false)
st.title("YBOT-Personalised Ai trainer for frontline workers")
st.subheader("Chat on organisational documents")

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


NEW_INDEX = "(New index)"
# Layout for the form 
options = get_index_list().names() + [NEW_INDEX]
st.session_state['index'] = options[0]
selection = st.selectbox("Select option", options=options)

# Just to show the selected option
if selection == NEW_INDEX:
    otherOption = st.text_input("Enter your other option...")
    if otherOption:
        selection = otherOption
        index_init(otherOption, 1536)
        st.info(f":white_check_mark: New index {otherOption} created! ")
        
st.session_state['index'] = selection
    

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


st.title("Media Chat on onboarding and new onboarding video")

uploaded_file = st.file_uploader("Select a video...", type=["mp4", "mov", "avi", "mp3", "mpeg", "mpga", "m4a", "wav", "webm"])

if uploaded_file is not None:
    video_path = uploaded_file.name

    with open(video_path, mode='wb') as f:
        f.write(uploaded_file.read())  # Save uploaded video to disk

    st.video(video_path)  # Display the uploaded video
    text = ''
    if uploaded_file.type in ["mp4", "mov", "avi"]:
        frames, text = video_to_text(video_path)    
    text += audio_to_text(video_path)
        
    st.write("Saarthi is analysing the video...")

    status = upload_file_to_pinecone(text, video_path, st.session_state['index'])

    if status == OK:
        st.write("Saarthi is ready to answer your questions")
    else:
        st.write(f"Error: {status}")
    
    


def main():
    # Add your main code here
  
    uploaded_file = st.file_uploader("Chat with your PDF file", type="pdf")
    
    if uploaded_file is not None:
        # To read file as bytes:
        
        name = uploaded_file.name
        st.write("File uploaded successfully!")
                
        raw_text = convert_pdf_to_txt_file(uploaded_file)
        
        st.write("Saarthi is analysing the fil")
        
        status = upload_file_to_pinecone(raw_text, name, st.session_state['index'])
        
        if status == OK:
            st.write("Saarthi is ready to answer your questions")
        else:
            st.write(f"Error: {status}")
        
if __name__ == "__main__":
    main()

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can i assist you"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = st.session_state.get('llm') or ChatOpenAI(model_name="gpt-4-turbo", temperature=0)

if 'buffer_memory' not in st.session_state:
    st.session_state['buffer_memory']=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'currently we dont know this but you can contact us at yogyabano.com'""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state['buffer_memory'], prompt=prompt_template, llm=llm, verbose=True)
from streamlit_mic_recorder import speech_to_text

def ask_query(query: str):
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            translated_text = translate_text('en', query)["translatedText"]
            # st.code(conversation_string)
            refined_query = query_refiner(translated_text, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query, st.session_state['index'])
            # print(context)  
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            translated_response = translate_text(iso_codes[st.session_state['language']], response)["translatedText"]
        st.session_state.requests.append(query)
        st.session_state.responses.append(translated_response)
        
def callback():
    if st.session_state.my_stt_output:
        st.write("You said: ", st.session_state.my_stt_output)
        ask_query(st.session_state.my_stt_output)



# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


with textcontainer:
    query = st.text_input("Query: ", key="input")
    ask_query(query)
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')




audio = speech_to_text(
    start_prompt="Saarthi is listening",
    stop_prompt="Saarthi has stopped listening",
    just_once=False,
    use_container_width=False,
    callback=callback,
    args=(),
    kwargs={},
    key='my_stt',
)


@st.cache_data
def convert_pdf_to_txt_file(path):
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
 
# if st.button("Ask a quiz with Saarthi"):
      
st.title("Take a quiz")

if st.button("Generate Quiz"):
    text = get_all_docs(st.session_state['index'])
    num = 10
    ans = generate_quiz(text, num)
    st.write("Here are the questions:")
    st.write(ans)st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        color: #4F6D7A;
        background: #FFFFFF
    }
    </style>
    """,
    unsafe_allow_html=True
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        color: #4F6D7A;
        background: #FFFFFF
    }
    </style>
    """,
    unsafe_allow_html=True
)
