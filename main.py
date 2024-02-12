from langchain.schema import ChatMessage
from langchain.chat_models import GigaChat
import streamlit as st

# Авторизация в сервисе GigaChat
# chat = GigaChat(credentials=st.secrets['GIGACHAT_API_KEY'])

st.title('Чат бот')

# Инициализируем модель LLM
#if 'llm' not in st.session_state:
#    st.session_state['llm'] = 'GigaChat:latest'

# Инициализируем историю чата, если сообщений еще не существует
messages = []
if messages not in st.session_state:
    st.session_state.messages = []

# Показываем все сообщения чата в сессии
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Инициализируем пользовательский ввод
promt = st.chat_input(placeholder='Введите сообщение')
if promt:
    # Показываем сообщение пользователя в "чат-контейнере"
    with st.chat_message('user'):
        st.markdown(promt)
    # Добавляем сообщение пользователя в историю
    st.session_state.messages.append({'role': 'user', 'content': promt})
    # Показываем сообщение ассистента в "чат-контейнере"
    # message_placeholder = st.empty()
    # full_response = ''


    # Ассистент-эхо
    response = f'Echo: {promt}'
    # Показываем ответ ассистента в "чат-контейнере"
    with st.chat_message('assistant'):
        st.markdown(response)
    # Добавляем ответ ассистента в историю
    st.session_state.messages.append({'role': 'assistant', 'content': response})