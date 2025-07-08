import streamlit as st
from rag_engine import ask_medical_question

st.set_page_config(page_title="MedCode AI")

st.title("MedCode AI")
st.markdown("Ask about medical procedures, diagnoses or billing codes:")

query = st.text_area("Enter your medical question:")

if st.button("Submit"):
    if query.strip() =="":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving information..."):
            answer = ask_medical_question(query)
            st.markdown("### GPT Answer")
            st.markdown(answer)