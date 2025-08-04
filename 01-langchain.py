from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

# prompt = ChatPromptTemplate.from_template("주제 {topic}에 대해 짧은 설명을 해주세요.")
# model = ChatOpenAI(model_name="gpt-4o-mini")
# chain = prompt | model | StrOutputParser()
# analysis_prompt = ChatPromptTemplate.from_template("이 대답을 영어로 번역해 주세요: {answer}")
# print(chain.invoke(["인플레이션", "더블딥"]))

# for token in chain.stream("더블딥"):
#     print(token, end="", flush=True)

# composed_chain_with_lambda = chain | (lambda input: {"answer": input}) | analysis_prompt | model | StrOutputParser()
# print(composed_chain_with_lambda.invoke({"topic": "더블딥"}))

chat = ChatOpenAI(model_name="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 금융 상담사입니다. 사용자에게 최선의 금융 조언을 제공합니다."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)
chain = prompt | chat
chat_history = ChatMessageHistory()
chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
print(
    chain_with_message_history.invoke(
        {"input": "저축을 늘리기 위해 무엇을 할 수 있나요?"}, {"configurable": {"session_id": "unused"}}
    ).content
)

print(
    chain_with_message_history.invoke(
        {"input": "방금 뭐라고 했나요?"}, {"configurable": {"session_id": "unused"}}
    ).content
)

# chat_history.add_user_message("저축을 늘리기 위해 무엇을 할 수 있나요?")
# chat_history.add_ai_message("저축 목표를 설정하고, 매달 자동 이체로 일정 금액을 저축하세요.")
# chat_history.add_user_message("방금 뭐라고 했나요?")
# ai_response = chain.invoke({"messages": chat_history.messages})
# print(ai_response.content)

# ai_msg = chain.invoke(
#     {
#         "messages": [
#             ("human", "저축을 늘리기 위해 무엇을 할 수 있나요?"),
#             ("ai", "저축 목표를 설정하고, 매달 자동 이체로 일정 금액을 저축하세요."),
#             ("human", "방금 뭐라고 했나요?"),
#         ]
#     }
# )
# print(ai_msg.content)
