import os
import requests
import logging
from pathlib import Path
import streamlit as st
import pandas as pd
from pypdf import PdfReader

from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.document_loaders import WebBaseLoader, WikipediaLoader

# for local files
CONTENT_DIR = os.path.dirname(__file__)

# Define functions to load different file types
def load_txt_file(file_path):
    """Load a single text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return Document(page_content=f.read(), metadata={'title': os.path.basename(file_path)})

def load_csv_file(file_path):
    """Load a single CSV file."""
    df = pd.read_csv(file_path)
    doc_text = df.to_csv(index=False)
    return Document(page_content=doc_text, metadata={'title': os.path.basename(file_path)})

def load_pdf_file(file_path):
    """Load a single PDF file."""
    docs = []
    pdf_reader = PdfReader(file_path)
    for num, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        doc = Document(page_content=page_text, metadata={'title': os.path.basename(file_path), 'page': (num + 1)})
        docs.append(doc)
    return docs

def load_md_file(file_path):
    """Load a single Markdown file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return Document(page_content=f.read(), metadata={'title': os.path.basename(file_path)})

def load_web_page(page_url):
    loader = WebBaseLoader(page_url)
    data = loader.load()
    return data

def load_online_pdf(pdf_url):
    loader = OnlinePDFLoader(pdf_url)
    data = loader.load()
    return data

def filename_from_url(url):
    filename = url.split("/")[-1]
    return filename

def get_wiki_docs(query, load_max_docs=2):
    wiki_loader = WikipediaLoader(query=query, load_max_docs=load_max_docs)
    docs = wiki_loader.load()
    return docs

def list_files(data_dir="./data", extensions=['.txt', '.csv', '.pdf', '.md']):
    """List all files in the given directory with the specified extensions."""
    paths = []
    for ext in extensions:
        paths.extend(Path(data_dir).rglob(f'*{ext}'))
    return [str(path) for path in paths]

def download_file(url, filename=None):
    response = requests.get(url)
    if not filename:
        filename = filename_from_url(url)

    full_path = os.path.join("./data", filename)

    with open(full_path, mode="wb") as f:
        f.write(response.content)
        download_path = os.path.realpath(f.name)
    return download_path


def load_files(data_dir="./data"):
    """Load all supported files from the given directory."""
    files = list_files(data_dir)
    docs = []
    for file_path in files:
        print(f"Loading {file_path}")
        try:
            if file_path.endswith('.txt'):
                docs.append(load_txt_file(file_path))
            elif file_path.endswith('.csv'):
                docs.append(load_csv_file(file_path))
            elif file_path.endswith('.pdf'):
                docs.extend(load_pdf_file(file_path))
            elif file_path.endswith('.md'):
                docs.append(load_md_file(file_path))
            else:
                logging.warning(f"Unsupported file type: {file_path}")
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            pass
    return docs

def get_document_text(uploaded_file, title=None):
    """Load content from a given file-like object."""
    docs = []
    fname = uploaded_file.name
    if not title:
        title = os.path.basename(fname)
    if fname.lower().endswith('pdf'):
        pdf_reader = PdfReader(uploaded_file)
        for num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            doc = Document(page_content=page_text, metadata={'title': title, 'page': (num + 1)})
            docs.append(doc)
    elif fname.lower().endswith('csv'):
        df = pd.read_csv(uploaded_file)
        doc_text = df.to_csv(index=False)
        docs.append(Document(page_content=doc_text, metadata={'title': title}))
    else:
        # assume text or markdown
        doc_text = uploaded_file.read().decode()
        docs.append(Document(page_content=doc_text, metadata={'title': title}))

    return docs

def main():
    st.title("File Viewer")

    # Sidebar for mode selection
    mode = st.sidebar.radio("Choose Mode", ["Offline", "Online"])

    if mode == "Offline":
        data_dir = "../data"
        files = list_files(data_dir)
        
        if not files:
            st.error(f"No files found in {data_dir}")
            return

        selected_file = st.selectbox("Select a file to view", [None] + files)  # Add None as the default option
        uploaded_file = st.file_uploader("Upload a file")

        if selected_file is not None:
            st.write(f"Displaying contents of: {selected_file}")
            docs = load_files(data_dir)
            for doc in docs:
                if doc.metadata['title'] == os.path.basename(selected_file):
                    if selected_file.endswith('.csv'):
                        st.dataframe(pd.read_csv(selected_file))
                    elif selected_file.endswith('.md'):
                        st.markdown(doc.page_content)
                    else:
                        st.text(doc.page_content)

        if uploaded_file is not None:
            st.write("Displaying contents of uploaded file:")
            try:
                docs = get_document_text(uploaded_file)
                for doc in docs:
                    if uploaded_file.name.endswith('.csv'):
                        st.dataframe(pd.read_csv(uploaded_file))
                    elif uploaded_file.name.endswith('.md'):
                        st.markdown(doc.page_content)
                    else:
                        st.text(doc.page_content)
            except Exception as e:
                st.error(f"Error loading uploaded file: {e}")

    elif mode == "Online":
        url = st.text_input("Enter URL of the file")
        file_type = st.selectbox("Select file type", ['PDF', 'Web Page'])
        if st.button("Load"):
            if url:
                if file_type == 'PDF':
                    try:
                        download_path = download_file(url)
                        docs = load_pdf_file(download_path)
                        for doc in docs:
                            st.text(doc.page_content)
                    except Exception as e:
                        st.error(f"Error loading PDF: {e}")
                elif file_type == 'Web Page':
                    try:
                        docs = load_web_page(url)
                        for doc in docs:
                            st.text(doc.page_content)
                    except Exception as e:
                        st.error(f"Error loading web page: {e}")


if __name__ == "__main__":
    main()