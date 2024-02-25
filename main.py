from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models.gigachat import GigaChat
# from langchain.chains import LLMChain
import streamlit as st

# Создаём экземпляр класса GigaChat и авторизируемся
giga = GigaChat(
    credentials=st.secrets['GIGACAHT_AUTH'],
    model='GigaChat:latest',
    verify_ssl_certs=False
)

st.title('Чат бот')

# Инициализируем историю чата, если сообщений ещё не существует
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Показываем все сообщения сессии в контейнере чата
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# Инициализируем пользовательский ввод
promt = st.chat_input(placeholder='Введите сообщение')
if promt:
    # Показываем сообщение пользователя в контейнере чата
    with st.chat_message('user'):
        st.markdown(promt)
    # Добавляем сообщение пользователя в историю
    st.session_state.messages.append(HumanMessage(content=promt))
    # Отправляем запрос и показываем ответ в контейнере чата
    response = giga(st.session_state.messages).content
    with st.chat_message('assistant'):
        st.markdown(response)
    # Добавляем ответ LLM в историю
    st.session_state.messages.append(AIMessage(response))