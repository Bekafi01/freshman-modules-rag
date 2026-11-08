from dotenv import load_dotenv
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def main():
    client = genai.Client()
    
    print("1. Fetching available Gemini models for your API key...")
    available_models = [
        m.name.replace("models/", "")
        for m in client.models.list()
        if "gemini" in m.name and "generateContent" in (m.supported_actions or [])
    ]
    
    for m in available_models:
        print(f"   - {m}")

    print("\n2. Testing all models to find every working endpoint...")
    working_models = []

    for model_name in available_models:
        # Skip specialized embedding/audio/image-only/robotics models
        if any(skip in model_name for skip in ["embedding", "tts", "image", "vision", "veo", "transcribe", "robotics", "computer-use"]):
            continue
            
        print(f"   Trying '{model_name}'...", end=" ", flush=True)
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                timeout=10,
                max_retries=0,
            )
            res = llm.invoke("Say 'setup complete!' in one word")
            print(f"SUCCESS! (Response: {res.text.strip()})")
            working_models.append(model_name)
        except Exception as e:
            err_msg = str(e).split("\n")[0][:90]
            print(f"FAILED ({err_msg})")

    print(f"\n3. Summary — Found {len(working_models)} working model(s):")
    for wm in working_models:
        print(f"   ✅ {wm}")

if __name__ == "__main__":
    main()