import os
import asyncio
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class RAGProcessor:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=self.api_key)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=self.api_key)
        self.vector_store_cache: Dict[str, Any] = {}
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def get_transcript_text(self, video_id: str, language: str = "en") -> str:
        try:
            languages = [language, 'en'] if language != 'en' else ['en']
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
            return " ".join(chunk["text"] for chunk in transcript_list)
        except (TranscriptsDisabled, NoTranscriptFound):
            try:
                transcript_list_obj = YouTubeTranscriptApi.list_transcripts(video_id)
                t = None
                try:
                    t = transcript_list_obj.find_manually_created_transcript(['en'])
                except:
                    try:
                        t = transcript_list_obj.find_generated_transcript(['en'])
                    except:
                        pass
                
                if not t:
                    try:
                        t = next(iter(transcript_list_obj))
                    except:
                        raise Exception("No transcripts found for this video.")
                
                return " ".join(chunk["text"] for chunk in t.fetch())
            except Exception as e:
                raise Exception(f"Could not retrieve transcript: {str(e)}")
        except Exception as e:
            raise Exception(f"Error fetching transcript: {str(e)}")

    def process_video(self, video_id: str, language: str = "en"):
        # Synchronous processing (CPU/Blocking IO)
        if video_id in self.vector_store_cache:
            return

        transcript_text = self.get_transcript_text(video_id, language)
        if not transcript_text:
            raise Exception("Transcript is empty")
            
        chunks = self.splitter.create_documents([transcript_text])
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store_cache[video_id] = vector_store

    async def answer_question(self, video_id: str, question: str, language: str = "en") -> str:
        if video_id not in self.vector_store_cache:
            # Offload blocking process_video to thread
            await asyncio.to_thread(self.process_video, video_id, language)
        
        vector_store = self.vector_store_cache[video_id]
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        template = """
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.
        Answer in the language requested: {language}.

        Context:
        {context}
        
        Question: {question}
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=['context', 'question', 'language']
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough(),
            'language': lambda x: language
        })

        chain = parallel_chain | prompt | self.llm | StrOutputParser()
        
        # Use ainvoke for non-blocking LLM call
        answer = await chain.ainvoke(question)
        return answer
