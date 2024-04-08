import streamlit as st
import os
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader

# Set Google API Key environment variable
os.environ["GOOGLE_API_KEY"] = "AIzaSyBMUY-nRzN0LNS1SToVGe4orj3MDmFmG3U"

# Function to summarize text from PDF using langchain
def summarize_pdf(text):
    llm = ChatGoogleGenerativeAI(temperature=0.3, model="gemini-pro")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
    chunks = text_splitter.create_documents([text])

    chain = load_summarize_chain(
        llm,
        chain_type='map_reduce',
        verbose=False
    )
    summary = chain.run(chunks)
    return summary

# Streamlit UI
def main():
    st.title("PDF Summarizer")

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Read the PDF file
        pdf_text = ""
        with st.spinner("Extracting text from PDF..."):
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text

        # Summarize the text
        with st.spinner("Summarizing text..."):
            summary = summarize_pdf(pdf_text)

        # Display the summary
        st.header("Summary:")
        st.write(summary)

if __name__ == "__main__":
    main()
