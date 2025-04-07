import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pandas as pd
import PyPDF2

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Load, chunk and index the contents of the blog.
# for web scrapping RAG datas
"""loader = WebBaseLoader(
    web_paths=("https://people.ieu.edu.tr/tr/senemkumovametin",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer("div", id = "short_cv")
    ),
)
docs = loader.load()
print(docs)"""
DocumentTXT = []
# for txt read
with open('2024-2025 Eğitim-Öğretim Yılı Hafızlık Eğitimi Kursları Uygulama Esasları.txt', 'r', encoding='utf-8') as file:
    content = file.read()

    documentTXT = [Document(page_content=content),
                   Document(page_content="Fatih Anamasli Senem Hocanın asistanı olarak görev yapmaktadır."),
                   Document(
                       page_content="İzmir ekonomi üniversitesinde 32 akts ders "
                                    "almak için minimum 3.0 ortalama ypamak gerekir."),
                   Document(
                       page_content="")]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
splits = text_splitter.split_documents(documentTXT)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)


if __name__ == "__main__":
    while True:
        do_continue = input("do you have more questions? (y/n)")
        if do_continue == "y":
            enter = input("what is your question")
            for chunk in rag_chain.stream(enter):
                print(chunk, end="", flush=True)
        else:
            break
