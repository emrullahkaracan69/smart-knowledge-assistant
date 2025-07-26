import os
import streamlit as st
from groq import Groq

st.write("Testing Groq API...")

# Get API key
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)

if api_key:
    st.success(f"API Key found: {api_key[:20]}...")
    
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Say hello"}],
            model="gemma-7b-it",
            max_tokens=50
        )
        st.success(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.error("No API key found!")