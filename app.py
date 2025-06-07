# app.py

import streamlit as st
from rag_engine import generate_response

st.set_page_config(page_title="SAP VIM Config Chatbot", layout="wide")
st.title("🤖 SAP VIM Configuration Chatbot")
st.markdown("Ask anything about SAP VIM — your assistant will respond using internal docs!")

query = st.text_area("📝 Ask a VIM-related question", height=120)

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🔍 Searching documents and generating response..."):
            response, chunks = generate_response(query)
        st.success("✅ Response ready!")

        st.markdown("### 💬 Answer")
        st.markdown(response)

        with st.expander("📄 Retrieved Context Chunks"):
            for i, c in enumerate(chunks, 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(c)