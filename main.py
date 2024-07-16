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
st.title("Yogyabano                  Empowering Skill Training with AI")
st.subheader("Saarthi - Your Personalized AI Trainer for Skill Training,")

#add a dropdown menu in streamlit 
st.sidebar.title("Choose a Language")
language = st.sidebar.selectbox(
    'Select a language',
    ['English', 'Hindi', 'Tamil', 'Telugu', 'Kannada', 'Malayalam', 'Bengali', 'Gujarati', 'Marathi', 'Punjabi', 'Odia', 'Assamese', 'Urdu', 'Sanskrit']
)


# st.sidebar.title("Choose an Index")
# Index = st.sidebar.selectbox(
#     'Select a Index',
#     ['fitter', '31may','uploaded-pdf' ]
# )



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


st.title("Media Upload and Processing")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mp3", "mpeg", "mpga", "m4a", "wav", "webm"])

if uploaded_file is not None:
    video_path = uploaded_file.name

    with open(video_path, mode='wb') as f:
        f.write(uploaded_file.read())  # Save uploaded video to disk

    st.video(video_path)  # Display the uploaded video
    text = ''
    if uploaded_file.type in ["mp4", "mov", "avi"]:
        frames, text = video_to_text(video_path)    
    text += audio_to_text(video_path)
        
    st.write("Uploading video to vector db...")

    status = upload_file_to_pinecone(text, video_path, st.session_state['index'])

    if status == OK:
        st.write("Video description embedded to index!")
    else:
        st.write(f"Error: {status}")
    
    



# class AudioProcessor(AudioProcessorBase):
#     def recv_annotated_audio(self, frames: av.AudioFrame):
#         # Here you can process the audio frames
#         return frames

# st.title("Microphone Button Integration in Streamlit")

# # Use the webrtc_streamer to create a microphone button
# webrtc_ctx = webrtc_streamer(
#     key="audio",
#     mode=WebRtcMode.SENDONLY,
#     audio_processor_factory=AudioProcessor,
#     media_stream_constraints={"audio": True}
# )

# if webrtc_ctx.state.playing:
#     st.write("Recording audio... Press the button again to stop.")
# else:
#     st.write("Press the button to start recording audio.")
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


audio = speech_to_text(
    start_prompt="Start recording",
    stop_prompt="Stop recording",
    just_once=False,
    use_container_width=False,
    callback=callback,
    args=(),
    kwargs={},
    key='my_stt',
)

# audio_bytes = audio["bytes"]
# sample_rate = audio["sample_rate"]  # Define the "sample_rate" variable
# sample_width = audio["sample_width"]
# id = audio["id"]

# audio_data = {
#     "bytes": audio_bytes,  # audio bytes mono signal, can be processed directly by st.audio
#     "sample_rate": sample_rate,  # depends on your browser's audio configuration
#     "sample_width": sample_width,  # 2
#     "format": "webm", # The file format of the audio sample
#     "id": id  # A unique timestamp identifier of the audio
# }

# Rest of the code...

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
 
def main():
    # Add your main code here
  
    uploaded_file = st.file_uploader("Chat with your PDF file", type="pdf")
    
    if uploaded_file is not None:
        # To read file as bytes:
        
        name = uploaded_file.name
        st.write("File uploaded successfully!")
                
        raw_text = convert_pdf_to_txt_file(uploaded_file)
        
        st.write("Uploading file to vector db...")
        
        status = upload_file_to_pinecone(raw_text, name, st.session_state['index'])
        
        if status == OK:
            st.write("File embedded to index!")
        else:
            st.write(f"Error: {status}")
        
if __name__ == "__main__":
    main()


# # Add a voice button here
# webrtc_ctx = webrtc_streamer(
#     key="audio",
#     mode=WebRtcMode.SENDONLY,
#     audio_processor_factory=None,  # Replace AudioProcessor with None
#     media_stream_constraints={"audio": True},
#     async_processing=True  # Add this argument to fix the issue
# )


# import ask_question  # Import the necessary module for asking a question

#vision if webrtc_ctx is not None:
#     if st.button("Transcribe and Ask"):
#         # Perform transcription and get the text
#         if webrtc_ctx.audio_processor is not None:
#             audio_text = transcribe_audio.transcribe_audio(webrtc_ctx.audio_processor.frames)
            
#             # Ask the question using the transcribed text
#             response = ask_question.ask_question(audio_text)
            
#             # Display the response
#             st.write("Response:", response)

# if webrtc_ctx is not None and webrtc_ctx.state.playing:
#     st.write("Recording audio... Press the button again to stop.")
# else:
#     st.write("Press the button to start recording audio.")
# # Add a button to transcribe the recorded audio and ask for queries

# import transcribe_audio  # Import the necessary module for transcribing audio
# import ask_question  # Import the necessary module for asking a question

# if st.button("Transcribe and Ask"):
#     if webrtc_ctx is not None:
#         # Perform transcription and get the text
#         audio_text = transcribe_audio.transcribe_audio(webrtc_ctx.audio_processor.frames)
        
#         # Ask the question using the transcribed text
#         response = ask_question.ask_question(audio_text)
        
#         # Display the response
#         st.write("Response:", response)







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




