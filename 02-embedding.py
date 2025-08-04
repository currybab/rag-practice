import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings

load_dotenv()

# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

# print(embeddings.embed_query("Hello world"))

data = [
    "주식 시장이 급등했어요",
    "시장 물가가 올랐어요",
    "전통 시장에는 다양한 물품들을 팔아요",
    "부동산 시장이 점점 더 복잡해지고 있어요",
    "저는 빠른 비트를 좋아해요",
    "최근 비트코인 가격이 많이 변동했어요",
]
df = pd.DataFrame(data, columns=["text"])
# print(df)


def get_embedding(text: str) -> list[float]:
    return embeddings.embed_query(text)


df["embedding"] = df.apply((lambda row: get_embedding(row.text)), axis=1)
# print(df)


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def return_answer_candidate(df, query):
    query_embedding = get_embedding(query)
    df["similarity"] = df.embedding.apply(lambda row: cos_sim(np.array(row), np.array(query_embedding)))
    return df.sort_values("similarity", ascending=False).head(3)


sim_result = return_answer_candidate(df, "과일 값이 비싸다")
print(sim_result)
