import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
import openai
from langchain import hub
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

# Step 1: Veri Yükleme
# CSV dosyasını okuyup DataFrame oluştur
file_path = "1-176_ps_browse_ps5.csv"
data = pd.read_csv(file_path)
# Chroma veritabanı için gerekli metinleri birleştir
texts = data["name:"] + " " + data["Genres:"] + " " + data["rating:"].astype(str) + "" + data["rating_count:"].astype(str)
metadatas = data.to_dict(orient="records")

# Embedding işlemi için OpenAIEmbeddings kullan
embeddings = OpenAIEmbeddings()

# Metinleri gömüye dönüştür ve vektör veritabanına ekle
embedded_texts = embeddings.embed_documents(texts.tolist())
# Step 2: Chroma Vektör Veritabanı Kurulumu
vector_db = Chroma(persist_directory="chroma_db", embedding_function=OpenAIEmbeddings)
vector_db.add_texts(texts=texts.tolist(), metadatas=metadatas, embeddings=embedded_texts)

# Verileri vektör veritabanına ekle
vector_db.add_texts(texts=texts.tolist(), metadatas=metadatas)

# Step 3: OpenAI LLM ve RetrievalQA Oluşturma
llm = OpenAI(temperature=0)
retriever = vector_db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Step 4: RAG Chatbot için Conversational Chain
chatbot_chain = ConversationalRetrievalChain(retriever=retriever, llm=llm)

# Step 5: Kullanıcıdan Soru Al ve Yanıtla
def main():
    print("PS5 Oyunları Analiz Chatbot'una Hoş Geldiniz!")
    print("Sorularınızla PS5 oyunları hakkında bilgi alabilirsiniz.")
    print("Çıkmak için 'exit' yazabilirsiniz.")

    while True:
        # Kullanıcı girişi al
        user_input = input("Soru: ")
        if user_input.lower() == "exit":
            print("Chatbot sonlandırıldı. Görüşmek üzere!")
            break

        # Soruyu işleme ve yanıt oluşturma
        response = chatbot_chain.run(user_input)
        print("Cevap:", response)

# Uygulamayı çalıştır
if __name__ == "__main__":
    main()
