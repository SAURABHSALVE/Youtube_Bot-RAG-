import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Setup page config
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .stButton > button {
        border-radius: 10px;
        width: 100%;
    }
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.bot {
        background-color: #475063;
    }
    .stChatMessage {
        background-color: rgba(0,0,0,0);
    }
</style>
""", unsafe_allow_html=True)

# Function to extract video ID from various YouTube URL formats
def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be" in url:
        return url.split("/")[-1]
    return url

# Function to format documents for the prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Main Application
def main():
    st.title("🎥 YouTube Chatbot")
    st.markdown("##### Chat with any YouTube video using RAG & LangChain")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Input
        api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key here.")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Language Selection
        language_code = st.text_input(
            "Language Code", 
            value="en",
            help="e.g., 'en' for English, 'es' for Spanish, 'hi' for Hindi"
        )
        
        # Reset Button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.vector_store = None
            st.session_state.current_video_id = None
            st.rerun()

    # Main Content Area
    col1, col2 = st.columns([3, 1])
    with col1:
        video_url = st.text_input("🔗 Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    
    # Process Video Logic
    if video_url and api_key:
        video_id = extract_video_id(video_url)
        
        # Check if we need to process a new video
        if "current_video_id" not in st.session_state or st.session_state.current_video_id != video_id:
            with st.spinner("🔄 Processing video transcript... This might take a moment."):
                try:
                    # 1. Fetch Transcript
                    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=[language_code]).to_raw_data()
                    transcript_text = " ".join([chunk["text"] for chunk in transcript_list])
                    
                    # 2. Split Text
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    docs = splitter.create_documents([transcript_text])
                    
                    # 3. Create Vector Store
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                    st.session_state.vector_store = FAISS.from_documents(docs, embeddings)
                    st.session_state.current_video_id = video_id
                    st.session_state.messages = [] # Reset chat for new video
                    st.success("✅ Video processed! You can now ask questions.")
                    
                except TranscriptsDisabled:
                    st.error("❌ Transcripts are disabled for this video.")
                except NoTranscriptFound:
                    st.error(f"❌ No transcript found for language '{language_code}'.")
                except Exception as e:
                    st.error(f"❌ Error occurred: {str(e)}")
                    # st.session_state.current_video_id = None # Keep the ID but maybe allow retry

    elif not api_key:
        st.info("👋 Hey! Please enter your OpenAI API Key in the sidebar to get started.")

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input and Response
    if st.session_state.get("vector_store"):
        if prompt := st.chat_input("Ask something about the video..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Thinking..."):
                    try:
                        retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})
                        
                        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
                        
                        template = """
                        You are a helpful assistant.
                        Answer ONLY from the provided transcript context.
                        If the context is insufficient, just say you don't know.
                        
                        Context:
                        {context}
                        
                        Question: {question}
                        """
                        
                        prompt_template = PromptTemplate(
                            template=template,
                            input_variables=['context', 'question']
                        )
                        
                        def format_docs(docs):
                            return "\n\n".join(doc.page_content for doc in docs)

                        # Retrieve and format context manually to display it
                        retrieved_docs = retriever.invoke(prompt)
                        context_text = format_docs(retrieved_docs)
                        
                        with st.expander("Debugging: Retrieved Context"):
                            st.write(retrieved_docs)

                        chain = (
                            {"context": lambda x: context_text, "question": RunnablePassthrough()}
                            | prompt_template
                            | llm
                            | StrOutputParser()
                        )
                        
                        response = chain.invoke(prompt)
                        message_placeholder.markdown(response)
                        
                        # Add assistant response to history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        message_placeholder.error(error_msg)

if __name__ == "__main__":
    main()
