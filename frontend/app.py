import streamlit as st
import requests
import json

st.set_page_config(page_title="Medical Q&A Assistant", page_icon="🏥", layout="wide")

API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000", help="URL of the API server")

st.title("🏥 Medical Question Answering Assistant")
st.markdown("Ask medical questions and get evidence-based answers powered by fine-tuned LLM")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    max_tokens = st.slider("Max Tokens", 64, 1024, 512)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05)
    use_rag = st.checkbox("Use RAG (Retrieval Augmented Generation)", value=False)
    
    st.divider()
    st.markdown("### Example Questions")
    examples = [
        "What are the symptoms of Type 2 diabetes?",
        "How does aspirin work to prevent heart attacks?",
        "What is the difference between COVID-19 and flu?"
    ]
    for example in examples:
        if st.button(example, key=example, use_container_width=True):
            st.session_state.question = example

# Main interface
question = st.text_area(
    "Your Medical Question:",
    value=st.session_state.get("question", ""),
    height=100,
    placeholder="e.g., What are the early signs of Alzheimer's disease?"
)

col1, col2 = st.columns([1, 4])
with col1:
    generate_btn = st.button("🔍 Get Answer", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.question = ""
        st.rerun()

if generate_btn and question:
    with st.spinner("Generating response..."):
        try:
            response = requests.post(
                f"{API_URL}/generate",
                json={
                    "prompt": f"### Question: {question}\n\n### Answer:",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "use_rag": use_rag
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                
                st.success("✅ Response Generated")
                st.markdown("### Answer:")
                st.write(data["text"])
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tokens Generated", data["tokens_generated"])
                with col2:
                    st.metric("Latency", f"{data['latency_ms']}ms")
                with col3:
                    st.metric("RAG Used", "Yes" if data.get("context_used") else "No")
                
                # Disclaimer
                st.warning("⚠️ **Medical Disclaimer:** This is an AI assistant for educational purposes. Always consult healthcare professionals for medical advice.")
            else:
                st.error(f"Error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to API server. Make sure the API is running at the specified URL.")
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The model might be taking too long to generate a response.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <p>Powered by Fine-Tuned Mistral-7B | Data Sources: PubMed, Medical Literature</p>
</div>
""", unsafe_allow_html=True)

