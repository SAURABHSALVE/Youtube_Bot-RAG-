import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# Setup page config
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #FF0000 0%, #2b313e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
    }
    
    .sub-header {
        text-align: center;
        color: #a0a0a0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* Input Fields Styling */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 1px solid #475063;
        background-color: #262730;
        color: #ffffff;
        padding: 12px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #FF0000;
        box-shadow: 0 0 0 1px #FF0000;
    }

    /* Button Styling */
    .stButton > button {
        border-radius: 12px;
        background-color: #FF0000;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #cc0000;
        transform: translateY(-2px);
    }

    /* Chat Message Bubbles */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .user-message {
        background-color: #2b313e;
        padding: 1rem;
        border-radius: 15px 15px 0 15px;
        margin-bottom: 1rem;
        border: 1px solid #3b4252;
    }
    
    .assistant-message {
        background-color: #1e2129;
        padding: 1rem;
        border-radius: 15px 15px 15px 0;
        margin-bottom: 1rem;
        border: 1px solid #2e3440;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161920;
        border-right: 1px solid #2e3440;
    }
    
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FF0000;
        margin-bottom: 1rem;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #262730;
        border-radius: 10px;
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
    # Header Section
    st.markdown('<h1 class="main-header">🎥 YouTube RAG Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Chat with any YouTube video instantly using the power of AI</p>', unsafe_allow_html=True)
    
    # Initialize session state for messages if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-header">⚙️ Configuration</div>', unsafe_allow_html=True)
        
        # API Key Check
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("⚠️ OPENAI_API_KEY not found in .env file.")
            st.info("Please create a .env file with your API key to proceed.")
        else:
            st.success("✅ API Key Loaded")
        
        st.markdown("---")
        
        # Language Selection
        languages = {
            "English": "en",
            "Hindi": "hi",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Japanese": "ja",
            "Russian": "ru",
            "Portuguese": "pt"
        }
        
        selected_language = st.selectbox(
            "Select Video Language",
            options=list(languages.keys()),
            help="Choose the language of the YouTube video."
        )
        language_code = languages[selected_language]
        
        st.markdown("---")
        
        # Reset Button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.vector_store = None
            st.session_state.current_video_id = None
            st.rerun()
            
        st.markdown("---")
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            1. Paste a YouTube URL in the input field.
            2. Wait for the transcript to be processed.
            3. Ask any question about the video's content!
            """)

    # Main Content Area
    col1, col2 = st.columns([4, 1])
    with col1:
        video_url = st.text_input("🔗 Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    
    # Process Video Logic
    if video_url and api_key:
        video_id = extract_video_id(video_url)
        
        # Check if we need to process a new video
        if "current_video_id" not in st.session_state or st.session_state.current_video_id != video_id:
            with st.status("🔄 Processing video...", expanded=True) as status:
                try:
                    st.write("📥 Fetching transcript...")
                    # 1. Fetch Transcript
                    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=[language_code]).to_raw_data()
                    transcript_text = " ".join([chunk["text"] for chunk in transcript_list])
                    
                    st.write("✂️ Splitting text...")
                    # 2. Split Text
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    docs = splitter.create_documents([transcript_text])
                    
                    st.write("🧠 Generating embeddings & vector store...")
                    # 3. Create Vector Store
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                    st.session_state.vector_store = FAISS.from_documents(docs, embeddings)
                    st.session_state.current_video_id = video_id
                    st.session_state.messages = [] # Reset chat for new video
                    
                    status.update(label="✅ Video processed successfully!", state="complete", expanded=False)
                    st.toast("Ready to chat!", icon="🎉")
                    
                except TranscriptsDisabled:
                    status.update(label="❌ Error", state="error")
                    st.error("Transcripts are disabled for this video.")
                except NoTranscriptFound:
                    status.update(label="❌ Error", state="error")
                    st.error(f"No transcript found for language '{language_code}'.")
                except Exception as e:
                    status.update(label="❌ Error", state="error")
                    st.error(f"Error occurred: {str(e)}")

    elif not api_key:
        st.info("👋 Hey! Please configure your OPENAI_API_KEY in the .env file to get started.")

    st.markdown("---")

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
                        retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})
                        
                        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
                        
                        # Helper to format chat history
                        def format_chat_history(messages):
                            history = []
                            for msg in messages:
                                if msg["role"] == "user":
                                    history.append(HumanMessage(content=msg["content"]))
                                elif msg["role"] == "assistant":
                                    history.append(AIMessage(content=msg["content"]))
                            return history

                        template = """
                        You are a helpful and intelligent assistant.
                        Your primary source of information is the provided transcript context from a YouTube video.
                        
                        Instructions:
                        1. Answer the question based on the Context provided below and the Chat History.
                        2. If the answer is NOT in the Context, you may use your general knowledge to answer, but you MUST mention that this information is not from the video.
                        3. Keep your language simple, clear, and easy to understand for a general audience.
                        4. Do not hallucinate. If you don't know the answer and it's not in the context or general knowledge, say you don't know.
                        5. be concise and to the point.
                        
                        Chat History:
                        {chat_history}
                        
                        Context:
                        {context}
                        
                        Question: {question}
                        """
                        
                        prompt_template = PromptTemplate(
                            template=template,
                            input_variables=['chat_history', 'context', 'question']
                        )
                        
                        # Retrieve and format context manually to display it
                        retrieved_docs = retriever.invoke(prompt)
                        context_text = format_docs(retrieved_docs)
                        
                        with st.expander("🔍 Debugging: Retrieved Context"):
                            st.write(retrieved_docs)

                        chain = (
                            {
                                "context": lambda x: context_text, 
                                "question": RunnablePassthrough(),
                                "chat_history": lambda x: format_chat_history(st.session_state.get("messages", [])[:-1])
                            }
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
