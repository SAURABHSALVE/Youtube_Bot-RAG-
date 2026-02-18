# 🎥 WatchLess - YouTube RAG Chatbot

**Chat with any YouTube video instantly using the power of AI.**  
WatchLess allows you to extract insights, summarize content, and ask specific questions about any YouTube video. Powered by **LangChain**, **OpenAI GPT-4o**, and **Streamlit**.

🔴 **Live Demo:** [https://watchless.streamlit.app/](https://watchless.streamlit.app/)

---

## 🚀 Key Features

- **🧠 RAG Technology**: Uses Retrieval-Augmented Generation to provide accurate answers based *only* on the video transcript.
- **💬 Smart Memory**: The chatbot remembers your previous questions, allowing for a natural conversation flow.
- **🌐 Multi-Language Support**: Works with videos in English, Hindi, Spanish, French, German, and more.
- **🎯 Precision & Context**: Retrieves specific segments (up to 20 chunks) for deep context understanding.
- **⚡ GPT-4o Powered**: Utilizes OpenAI's latest flagship model for superior reasoning and summarization.
- **🎨 Modern UI**: Features a clean, dark-themed responsive interface.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **AI Framework**: [LangChain](https://www.langchain.com/)
- **LLM**: OpenAI GPT-4o
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Data Source**: `youtube-transcript-api`

---

## 📦 Installation & Setup

Follow these steps to run the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/SAURABHSALVE/Youtube_Bot-RAG-.git
cd Youtube_Bot-RAG-
```

### 2. Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Create a `.env` file in the project root directory and add your OpenAI API key:
```env
OPENAI_API_KEY=sk-proj-your-api-key-here...
```

### 5. Run the Application
```bash
streamlit run app.py
```

---

## 💡 How to Use
1. **Enter URL**: Paste any YouTube video link.
2. **Select Language**: Choose the language of the video (e.g., English, Hindi).
3. **Wait for Processing**: The app will fetch the transcript and generate embeddings.
4. **Start Chatting**: Ask "Who is the speaker?", "Summarize the key points", or anything else!

---

## 🤝 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

---
*Built with ❤️ by Saurabh Salve*
