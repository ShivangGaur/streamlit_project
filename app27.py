from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

def read_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text = "\n".join([page.page_content for page in pages])
    return text

def answer_question(pdf_text, question):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_text(pdf_text, embeddings)
    qa_pipeline = HuggingFacePipeline(model="MBZUAI/LaMini-T5-738M")
    retriever = db.as_retriever()
    qa = RetrievalQA(llm=qa_pipeline, retriever=retriever)
    answer = qa(question)
    return answer['result']

if __name__ == "__main__":
    pdf_path = "path_to_your_pdf_file.pdf"
    user_question = input("Ask a question: ")
    pdf_text = read_pdf(pdf_path)
    answer = answer_question(pdf_text, user_question)
    print("Answer:", answer)
