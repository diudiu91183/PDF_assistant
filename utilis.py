"""
远航的AI学习之旅
utilis -

Author: Administrator --Mike Yang
Date: 2026/2/23
"""

from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


def qa_tool(openai_api_key, memory, uploaded_file, question):
    # model = ChatOpenAI(model="qwen1.5-72b-chat",
    #                    openai_api_key="sk-7b05a65d0ec844d2878a153e50c8d92c",
    #                    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = ChatOpenAI(model="qwen1.5-72b-chat",
                       openai_api_key="sk-7b05a65d0ec844d2878a153e50c8d92c",
                       openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1")
    file_content = uploaded_file.read()
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(file_content)
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", '。', '！', "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)
    # embedding_model = OpenAIEmbeddings()
    embedding_model = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",  # 专门针对中文优化的模型
        model_kwargs={'device': 'cpu'},  # 使用 CPU，如果有 GPU 可以改为 'cuda'
        encode_kwargs={
            'normalize_embeddings': True,  # 归一化向量
            'batch_size': 32  # 批处理大小
        }
    )
    db = FAISS.from_documents(texts, embedding_model)
    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )
    reponse = qa.invoke({'chat_history': memory, 'question': question})
    return reponse
