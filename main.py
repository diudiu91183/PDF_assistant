"""
远航的AI学习之旅
main -

Author: Administrator --Mike Yang
Date: 2026/2/23
"""
import streamlit as st
from utilis import qa_tool
from langchain.memory import ConversationBufferMemory

st.title('智能PDF问答工具')
with st.sidebar:
    openai_api_key = st.text_input("请输入Qwen API 密钥：", "sk-7b05a65d0ec844d2878a153e50c8d92c", type="password")
    st.markdown("[获取Qwen API 密钥](https://bailian.console.aliyun.com/cn-beijing/?tab=model#/api-key)")
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )
upload_file = st.file_uploader("请上传文件：", type="pdf")
question = st.text_input('对PDF进行提问：', disabled=not upload_file)

if upload_file and question and not openai_api_key:
    st.info('请输入Qwen API 密钥！')
if upload_file and question and openai_api_key:
    with st.spinner('AI正在思考中，请稍等。。。'):
        response = qa_tool(openai_api_key, st.session_state["memory"], upload_file, question)
    st.write('###回答：')
    st.write(response['answer'])
    st.session_state['chat_history'] = response['chat_history']

if 'chat_history' in st.session_state:
    with st.expander('历史消息'):
        for i in range(0, len(st.session_state['chat_history']), 2):
            human_message = st.session_state['chat_history'][i]
            ai_message = st.session_state['chat_history'][i + 1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(st.session_state['chat_history']) - 2:
                st.divider()
