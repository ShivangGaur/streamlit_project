from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(pdf_path):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    raw_text = get_pdf_text(pdf_path)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    print("PDF loaded and processed successfully.")
    while True:
        user_question = input("Ask a Question from the PDF Files (type 'exit' to quit): ")
        if user_question.lower() == "exit":
            break
        chain = get_conversational_chain()
        response = chain({"input_documents": text_chunks, "question": user_question}, return_only_outputs=True)
        print("Reply: ", response["output_text"])

def main():
    print("Welcome to Chat with PDF using Gemini💁")
    pdf_path = input("Enter the path of the PDF file: ")
    user_input(pdf_path)

if __name__ == "__main__":
    main()
