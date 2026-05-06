import google.generativeai as genai

# 🔑 Replace with your API key
API_KEY = "AIzaSyBvo6p9LBruiBYDOFygIbDlw0RXe147xB4"

def check_models():
    try:
        genai.configure(api_key=API_KEY)

        print("Fetching available Gemini models...\n")

        models = genai.list_models()

        usable_models = []

        for model in models:
            # Check if model supports content generation
            if "generateContent" in model.supported_generation_methods:
                usable_models.append(model.name)

        if not usable_models:
            print("❌ No usable models found for your API key.")
            return

        print("✅ Usable Gemini Models:\n")
        for m in usable_models:
            print(f"👉 {m}")

    except Exception as e:
        print("❌ Error:", str(e))


if __name__ == "__main__":
    check_models()