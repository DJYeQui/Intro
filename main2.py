import csv

import chainlit as cl
import bs4
from langchain import hub
from langchain.document_loaders.parsers.html import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import langchain_experimental
from dotenv import load_dotenv

load_dotenv()

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")


def web_search(url) -> Document:
    # for web scrapping RAG datas
    url = "https://people.ieu.edu.tr/tr/senemkumovametin"
    loader = WebBaseLoader(
        web_paths=(url),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer("div", id = "short_cv")
        ),
    )
    return Document(page_content=str(loader.load()))  # THIS FUNCTION HAS NOT TESTED YET

def csv_file_reader(path) -> Document:
    with open(path, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            print(row)

# Loading content from a text file
with open('RAGInfo.txt', 'r', encoding='utf-8') as file:
    content = file.read()
    documentTXT = [
        Document(page_content=content),
        Document(page_content="Fatih Anamasli Senem Hocanın asistanı olarak görev yapmaktadır."),
        Document(page_content="İzmir ekonomi öğrencisidir. Başarılı bir yazılımcı ve game designerdır. "
                              "Geçmişinde girişimcilik maceraları olmuştur ve şirketlerde ve vc lerde çalışmıltır."),
        Document(page_content="İzmir ekonomi üniversitesinde 32 akts ders almak için minimum 3.0 ortalama ypamak gerekir.")
    ]

# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
splits = text_splitter.split_documents(documentTXT)

# Create a Chroma vectorstore from the documents
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Create a retriever
retriever = vectorstore.as_retriever()

# Load the RAG prompt template from the hub
prompt = hub.pull("rlm/rag-prompt")


# Function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Create a RAG chain
rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)


# Chainlit application setup
@cl.on_message
async def handle_message(message: cl.message.Message):
    # Extract the text content from the Chainlit message object
    user_input = message.content
    # Log the input message for debugging
    print(f"Received message: {user_input}")

    # Try to get the response from the RAG chain
    try:
        response = ""
        # Get the result from the chain (without async for, use regular for)
        result = rag_chain.invoke(user_input)

        # Iterate over the chunks and build the response
        for chunk in result:
            response += chunk
            print(f"Chunk received: {chunk}")  # Debug log

        # Send the response back to the user
        if response:
            await cl.Message(content=response).send()
        else:
            await cl.Message(content="No relevant information found.").send()

    except Exception as e:
        print(f"Error: {e}")
        await cl.Message(content="An error occurred while processing your request.").send()
