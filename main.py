import streamlit as st
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models.gigachat import GigaChat
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings.gigachat import GigaChatEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
from io import StringIO

# Создаём экземпляр класса GigaChat с авторизацией
giga = GigaChat(
    credentials=st.secrets['GIGACAHT_AUTH'],
    model='GigaChat:latest',
    verify_ssl_certs=False,
    stream=True
)

# Создаём экземпляр класса GigaChatEmbeddings с авторизацией
giga_embeddings = GigaChatEmbeddings(
    credentials=st.secrets['GIGACAHT_AUTH'],
    model='GigaChat:latest',
    verify_ssl_certs=False
)


def get_txt_text(file):
    # Конвертируем файл в строку и записываем её в переменную
    text = StringIO(file.getvalue().decode('utf-8')).read()
    print(text[0:51])
    return text


def get_text_chunks(text):
    # Инициализируем сплиттер и нарезаем текст на чанки
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    print(len(chunks))
    return chunks


def get_vector_store(text_chunks):
    # Создаём векторную базу данных эмбеддингов
    embeddings = giga_embeddings
    # Инициализируем векторное хранилище с использованием библиотеки FAISS,
    # которая создаёт индексы для быстрого поиска по векторам текста
    vectore_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )
    # Сохраняем полученную бд локально под именем 'faiss_index'
    vectore_store.save_local('faiss_index')
    vectordb = FAISS.load_local('faiss_index', embeddings)
    return vectordb


def get_relevant_chunks(vdb, query):
    search_results = vdb.similarity_search(query, k=3)
    concut_result = '/n '.join([result.page_content for result in search_results])
    return concut_result


st.title('Чат-бот с RAG')

# Инициализируем загрузчик файлов
uploaded_file = st.file_uploader(
    label='Загрузите файл и нажмите кнопку "Обработать"',
    type='txt'
)

# Обрабатываем файл, если он загружен и нажата кнопка
if st.button('Обработать', key='process_button'):
    with st.spinner('Идёт обработка файла...'):
        raw_text = get_txt_text(file=uploaded_file)
        text_chunks = get_text_chunks(text=raw_text)
        vectordb = get_vector_store(text_chunks=text_chunks)
        st.session_state['vectordb'] = vectordb
        st.success('Готово')

# Инициализируем историю чата
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Показываем все сообщения сессии, кроме системных, в контейнере чата
for message in st.session_state.messages:
    if message['role'] != 'system':
        with st.chat_message(message['role']):
            st.markdown(message['content'])


# # Показываем все сообщения сессии в контейнере чата
# for message in st.session_state.messages:
#     with st.chat_message(message.type):
#         st.markdown(message.content)

# Задаём шаблон промта
promt_template = '''
     Ты - ИИ помощник, который отвечает на вопросы пользователей.
     Отвечай на вопрос кратко и по существу, опирайся на предоставленный контекст.
     Если ответа нет в предоставленном контексте, скажи: "Ответ недоступен в контексте", не пиши неправильный ответ.
     Контекст: {context}
     '''

# Инициализируем пользовательский ввод
user_input = st.chat_input(placeholder='Задайте вопрос')

if user_input:
    doc_db = st.session_state.get('vectordb', None)
    if not doc_db:
        pass
    relevant_search = get_relevant_chunks(vdb=doc_db, query=user_input)
    context = promt_template.format(context=relevant_search)
    # Обогащаем промт результатами поиска по бд
    st.session_state.messages.insert(0, SystemMessage(content=context))
    # Добавляем сообщение пользователя в историю
    st.session_state.messages.append(HumanMessage(content=user_input))
    # Показываем сообщение пользователя в контейнере чата
    with st.chat_message('user'):
        st.markdown(user_input)
    # Отправляем запрос LLM
    response = giga(st.session_state.messages)
    # Добавляем ответ LLM в историю
    st.session_state.messages.append(AIMessage(response.content))
    # Показываем ответ LLM в контейнере чата
    with st.chat_message('assistant'):
        st.markdown(response.content)
