from dotenv import load_dotenv
from langchain.memory import ChatMessageHistory
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

loader = PyPDFLoader("./2024_kb_real_estate_report.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print("number of chunks: ", len(chunks))

embedding_function = OpenAIEmbeddings()
persist_directory = "./db"
vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_function, persist_directory=persist_directory)
print("number of documents: ", vectorstore._collection.count())

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

template = """당신은 KB 부동산 보고서 전문가입니다. 다음 정보를 바탕으로 사용자의 질문에 답변해주세요.
컨텍스트: {context}
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ]
)
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


def format_docs(docs) -> str:
    return "\n\n".join([doc.page_content for doc in docs])


chain = (
    RunnablePassthrough.assign(context=lambda x: format_docs(retriever.invoke(x["question"])))
    | prompt
    | model
    | StrOutputParser()
)

chat_history = ChatMessageHistory()
chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: chat_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)


def chain_with_bot():
    session_id = "user_session"
    print("KB 부동산 보고서 챗봇입니다. 질문해 주세요. (종료하려면 quit 입력)")
    while True:
        user_input = input("User: ")
        if user_input.lower().strip() == "quit":
            break

        response = chain_with_message_history.invoke(
            {"question": user_input}, {"configurable": {"session_id": session_id}}
        )
        print("Bot: ", response)


chain_with_bot()
