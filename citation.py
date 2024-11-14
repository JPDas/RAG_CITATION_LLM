import os
import pdfplumber
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from IPython.display import display, Markdown, Latex
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_KEY")

def read_pdf_page_by_page(pdf_path):
    """Reads a PDF page by page and prints the text.

    Args:
        pdf_path (str): Path to the PDF file.
    """

    texts = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            texts += page.extract_text()
            
            # texts.append({"page_number": page_num+1, "text": text})

    return texts

def create_vector_store(texts):
    """Creates a vector store from a PDF file."""

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    docs = text_splitter.split_text(texts)
    
    # Create embeddings
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    metadata=[{"source": text} for text in docs]
    # Create the vector store
    vectorstore = FAISS.from_texts(docs, embeddings, metadatas=metadata)
    return vectorstore

def print_result(result):
    output_text = f"""### Question:
    {result['input']}
    ### Answer:
    {result['answer']}
    ### Sources:
    {result['context']}
    ### All relevant sources:
    {' '.join(list(set([doc.metadata['source'] for doc in result['context']])))}
    """
    return(output_text)


pdf_path = r"D:\LLM_Experiments\RAG_CITATION\Introduction to Machine Learning with Python ( PDFDrive.com )-min.pdf"

texts = read_pdf_page_by_page(pdf_path)

vector_store = create_vector_store(texts)

llm = ChatOpenAI(temperature = 0.0, model="gpt-4o-mini",  api_key=OPENAI_API_KEY)

CITATION_QA_TEMPLATE = (
    "Please provide an answer based solely on the provided sources. "
    "When referencing information from a source, "
    "cite the appropriate source(s) using their corresponding numbers. "
    "Every answer should include at least one source citation. "
    "Only cite a source when you are explicitly referencing it. "
    "If none of the sources are helpful, you should indicate that. "
    "For example:\n"
    "Source 1:\n"
    "The sky is red in the evening and blue in the morning.\n"
    "Source 2:\n"
    "Water is wet when the sky is red.\n"
    "Query: When is water wet?\n"
    "Answer: Water will be wet when the sky is red [2], "
    "which occurs in the evening [1].\n"
    "Now it's your turn. Below are several numbered sources of information:"
    "\n------\n"
    "{context}"
    "\n------\n"
    "Query: {input}\n"
    "Answer: "
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CITATION_QA_TEMPLATE),
        ("human", "{input}"),
    ]
)

retriever=vector_store.as_retriever(search_kwargs={"k":3})
query = "What is principal component analysis?"

question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

result = chain.invoke({"input": query})

print(result['answer'])
display(Markdown(result['answer']))