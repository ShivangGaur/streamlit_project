import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings

def read_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text = "\n".join([page.page_content for page in pages])
    return text

def answer_question(pdf_text, question):
    embeddings = HuggingFaceEmbeddings(model_name="google/bigbird-roberta-base")
    retriever = embeddings.as_retriever()
    qa = RetrievalQA(retriever)
    answer = qa({"context": pdf_text, "question": question})
    return answer['result']

def main():
    st.title("Chat with PDF")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file is not None:
        user_question = st.text_input("Ask a question")
        if st.button("Get Answer"):
            pdf_text = read_pdf(uploaded_file)
            answer = answer_question(pdf_text, user_question)
            st.write("Answer:", answer)

if __name__ == "__main__":
    main()
