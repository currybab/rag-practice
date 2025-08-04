import base64
import os
import uuid
from base64 import b64decode
from operator import itemgetter

import nltk
from dotenv import load_dotenv
from langchain.retrievers import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.stores import InMemoryStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from unstructured.partition.pdf import partition_pdf

load_dotenv()
fpath = "."
fname = "sample.pdf"

# download nltk
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")

# extract element from pdf
raw_pdf_elements = partition_pdf(
    filename=os.path.join(fpath, fname),
    extract_images_in_pdf=True,
    infer_table_structure=True,
    chunking_strategy="by_title",
    extract_image_block_output_dir=fpath,
)

# extract table and text elements
tables = []
texts = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        tables.append(str(element))  # add table element
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        texts.append(str(element))  # add text element

# print(tables[0])
# print(texts[0])
# print(len(tables))
# print(len(texts))

# create prompt
prompt_text = """당신은 표와 텍스트를 요약하여 검색할 수 있도록 돕는 역할을 맡은 어시스턴트입니다.
이 요약은 임베딩되어 원본 텍스트나 표 요소를 검색하는 데 사용될 것입니다.
표 또는 텍스트에 대한 간결한 요약을 제공하여 검색에 최적화된 형태로 만들어 주세요. 
표 또는 텍스트: {element} """
prompt = ChatPromptTemplate.from_template(prompt_text)

# text summarize chain
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

# text & table summarize
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

print(text_summaries[0])
print(table_summaries[0])


def encode_image(image_path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


img_base64_list = []

for img_file in sorted(os.listdir(fpath)):
    if img_file.endswith(".jpg"):
        img_base64_list.append(encode_image(os.path.join(fpath, img_file)))


def image_summarize(img_base64: str) -> str:
    chat = ChatOpenAI(model_name="gpt-4o", max_tokens=1024)
    prompt = """
    당신은 이미지를 요약하여 검색을 위해 사용할 수 있도록 돕는 어시스턴트입니다.
    이 요약은 임베딩되어 원본 이미지를 검색하는데 사용됩니다.
    이미지 검색에 최적화된 간결한 요약을 작성하세요.
    """
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                ]
            )
        ]
    )
    return msg.content


images_summaries = []
for img_base64 in img_base64_list:
    images_summaries.append(image_summarize(img_base64))

print(images_summaries[0])

vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=OpenAIEmbeddings())
docstore = InMemoryStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key=id_key,
)

# save source text data
doc_ids = [str(uuid.uuid4()) for _ in texts]
retriever.docstore.mset(list(zip(doc_ids, texts, strict=False)))

# save source table data
table_ids = [str(uuid.uuid4()) for _ in tables]
retriever.docstore.mset(list(zip(table_ids, tables, strict=False)))

# save source image data
img_ids = [str(uuid.uuid4()) for _ in img_base64_list]
retriever.docstore.mset(list(zip(img_ids, img_base64_list, strict=False)))

# save text summary vector
summary_texts = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(text_summaries)]
retriever.vectorstore.add_documents(summary_texts)

# save table summary vector
summary_tables = [Document(page_content=s, metadata={id_key: table_ids[i]}) for i, s in enumerate(table_summaries)]
retriever.vectorstore.add_documents(summary_tables)

# save image summary vector
summary_images = [Document(page_content=s, metadata={id_key: img_ids[i]}) for i, s in enumerate(images_summaries)]
retriever.vectorstore.add_documents(summary_images)


docs = retriever.invoke("말라리아 군집 사례는 어떤가요?")
print(len(docs))


def split_image_text_types(docs):
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception:
            text.append(doc)
    return {
        "images": b64,
        "texts": text,
    }


docs_by_type = split_image_text_types(docs)
print(len(docs_by_type["images"]))
print(len(docs_by_type["texts"]))


def prompt_func(dict):
    format_texts = "\n".join(dict["context"]["texts"])
    text = f"""
    다음 문맥에만 기반하여 질문에 답하세요. 문맥에는 텍스트, 표, 그리고 아래 이미지가 포함될 수 있습니다.
    질문: {dict["question"]}
    
    텍스트와 표:
    {format_texts}
    """
    prompt = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": text,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{dict['context']['images'][0]}",
                    },
                },
            ]
        )
    ]
    return prompt


model = ChatOpenAI(model_name="gpt-4o", temperature=0, max_tokens=1024)

chain = (
    {"context": retriever | RunnableLambda(split_image_text_types), "question": RunnablePassthrough()}
    | RunnableLambda(prompt_func)
    | model
    | StrOutputParser()
)

print(chain.invoke("말라리아 군집 사례는 어떤가요?"))
