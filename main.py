from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    # Test Google Gemini
    llm_gemini = ChatGoogleGenerativeAI(model="gemini-4-argon", temperature=0)
    response_gemini = llm_gemini.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatGoogleGenerativeAI: {response_gemini.text}")

    print("Setup complete!")

if __name__ == "__main__":
    main()