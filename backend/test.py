import google.generativeai as genai

try:
    # --- PASTE YOUR API KEY HERE ---
    api_key = "AIzaSyA8SV1D2kKlYR6JrCNqFRnY-JkZfdnwyI8" 

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')

    print("Model loaded. Generating content...")
    response = model.generate_content("Tell me a fun fact about the planet Mars.")

    print("\n--- RESPONSE ---")
    print(response.text)
    print("\n✅ Success! Your API key and installation are working correctly.")

except Exception as e:
    print(f"❌ An error occurred: {e}")