from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

loader = PyPDFLoader("./2024_kb_real_estate_report.pdf")
pages = loader.load()
print("number of chunks: ", len(pages))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(pages)
print("number of splitted chunks: ", len(splits))

chunk_lengths = [len(chunk.page_content) for chunk in splits]
max_length = max(chunk_lengths)
min_length = min(chunk_lengths)
print("max length: ", max_length)
print("min length: ", min_length)
print("average length: ", sum(chunk_lengths) / len(chunk_lengths))

# embedding_function = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
embedding_function = OpenAIEmbeddings()

persist_directory = "./db"
vectordb = Chroma.from_documents(documents=splits, embedding=embedding_function, persist_directory=persist_directory)
print("number of documents: ", vectordb._collection.count())

question = "수도권 주택 매매 전망"
top_three_docs = vectordb.similarity_search(question, k=3)
for i, doc in enumerate(top_three_docs, 1):
    print(f"{i}. {doc.page_content[:150]}")
    print(f"meta data: {doc.metadata}")
    print("-" * 20)


top_three_docs = vectordb.similarity_search_with_relevance_scores(question, k=3)
for i, doc in enumerate(top_three_docs, 1):
    print(f"{i}. {doc[0].page_content[:150]}")
    print(f"meta data: {doc[0].metadata}")
    print(f"score: {doc[1]}")
    print("-" * 20)
