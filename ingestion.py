from dotenv import load_dotenv
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
docs_splits = text_splitter.split_documents(docs_list)

embedding= GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    type="embedding",
)

# Una vez creado el vectorstore, no es necesario volver a crearlo. Descomentar la siguiente línea solo la primera vez.
# Esto crea la carpeta .chroma en el directorio actual.
# vectorstore = Chroma.from_documents(
#     documents=docs_splits,
#     embedding=embedding,
#     collection_name="rag-chroma",
#     persist_directory="./.chroma",
# )

retriever = Chroma(
    embedding_function=embedding,
    collection_name="rag-chroma",
    persist_directory="./.chroma",
).as_retriever()
