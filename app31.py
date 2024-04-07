import streamlit as st
import fitz  # PyMuPDF
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from heapq import nlargest

# Function to summarize text using TF-IDF algorithm
def summarize_text(text, num_sentences=5):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in sentences if word.lower() not in stop_words]
    
    # Compute word frequency
    freq_dist = FreqDist(words)
    
    # Rank sentences based on word frequency
    ranking = {}
    for i, sentence in enumerate(sentences):
        for word in nltk.word_tokenize(sentence.lower()):
            if word in freq_dist:
                if i in ranking:
                    ranking[i] += freq_dist[word]
                else:
                    ranking[i] = freq_dist[word]

    # Get top sentences
    top_sentences = nlargest(num_sentences, ranking, key=ranking.get)
    summary = [sentences[j] for j in sorted(top_sentences)]

    return ' '.join(summary)

# Streamlit UI
def main():
    st.title("PDF Summarizer")

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Read the PDF file
        pdf_text = ""
        with st.spinner("Extracting text..."):
            doc = fitz.open(uploaded_file)
            for page in doc:
                pdf_text += page.get_text()

        # Summarize the text
        with st.spinner("Summarizing text..."):
            summary = summarize_text(pdf_text)

        # Display the summary
        st.header("Summary:")
        st.write(summary)

if __name__ == "__main__":
    main()
