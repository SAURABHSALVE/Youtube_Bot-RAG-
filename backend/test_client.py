import requests
import sys

BASE_URL = "http://localhost:8000"

def get_video_id(url):
    """Extract video ID from URL"""
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return url

def main():
    print("🎥 YouTube Chatbot CLI Client")
    print("-----------------------------")
    
    video_url = input("Enter YouTube Video URL (or ID): ").strip()
    video_id = get_video_id(video_url)
    language = input("Enter language code (default 'en'): ").strip() or "en"
    
    print(f"\n⏳ Processing video {video_id}...")
    
    # 1. Process Video
    try:
        response = requests.post(f"{BASE_URL}/api/process", json={
            "video_id": video_id,
            "language": language
        })
        response.raise_for_status()
        print("✅ Video processed successfully!")
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        try: 
            print(f"Server response: {response.text}")
        except: pass
        return

    # 2. Chat Loop
    print("\n💬 Chat with the video (type 'exit' to quit)")
    print("-------------------------------------------")
    
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ['exit', 'quit', 'q']:
            break
        
        if not question:
            continue
            
        try:
            response = requests.post(f"{BASE_URL}/api/ask", json={
                "video_id": video_id,
                "question": question,
                "language": language
            })
            
            data = response.json()
            
            if response.status_code == 200:
                if data.get("error"):
                    print(f"⚠️  Warning: {data['error']}")
                
                if data.get("answer"):
                    print(f"🤖 Bot: {data['answer']}")
                else:
                    print("🤖 Bot: [No answer returned]")
            else:
                print(f"❌ Error: {data.get('detail', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    main()
