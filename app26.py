import streamlit as st

def upload_pdf():
    st.sidebar.title("Upload PDF File")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        st.sidebar.success("PDF file uploaded successfully!")
        st.sidebar.write(uploaded_file.name)
        
        # Optionally, you can read and display the contents of the PDF file
        pdf_contents = uploaded_file.read()
        st.write(pdf_contents)

if __name__ == "__main__":
    upload_pdf()
