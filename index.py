import streamlit as st
import chromadb
import openai
import os
from openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import TextNode
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb import EmbeddingFunction
from docx import Document
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Настройка модели и клиент Chroma
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)

class LlamaIndexEmbeddingAdapter(EmbeddingFunction):
    def __init__(self, ef: OpenAIEmbedding):
        self.ef = ef

    def __call__(self, input):
        return [
            node.embedding for node in self.ef([TextNode(text=doc) for doc in input])
        ]

chroma_client = chromadb.PersistentClient(
    path="files_embeddded",
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

chroma_collection = chroma_client.get_or_create_collection(
    "files_embeddded",
    embedding_function=LlamaIndexEmbeddingAdapter(Settings.embed_model),
)

openai.api_key = OPENAI_API_KEY
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def extract_text_from_file(uploaded_file):
    """Извлечение текста из файла Word или PDF."""
    if uploaded_file.name.endswith('.docx'):
        doc = Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif uploaded_file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        return "".join([page.extract_text() for page in pdf_reader.pages])
    else:
        st.error("Формат файла не поддерживается. Загрузите .docx или .pdf.")
        return ""

def get_similar_cases(text, max_distance=1.5):
    embedding = Settings.embed_model.get_text_embedding(text)
    results = chroma_collection.query(
        query_embeddings=[embedding],
        n_results=5
    )
    filtered_results = []
    for number, distance in enumerate(results["distances"][0]):
        if distance < max_distance:
            filtered_results.append(results["documents"][0][number])
    print(filtered_results)
    print(results)
    messages = [
        {
            "role": "user",
            "content": f"""
                    Твоя задача — анализировать уголовные дела, содержащие статью 218 УК РК (отмывание денег), на основе запроса клиента.
                    
                    Запрос клиента: {text}
                    
                    Действия, которые необходимо выполнить:
                    
                    Анализ уголовных дел:
                    
                    Найди уголовные дела, связанные со статьей 218 УК РК.
                    Классифицируй дела по предикатным преступлениям (например, мошенничество, коррупция, налоговые преступления).
                    Идентификация методов отмывания:
                    
                    Определи методы легализации незаконных доходов, включая:
                    Использование платежных карт;
                    Оформление фиктивных кредитов;
                    Создание и руководство финансовыми пирамидами;
                    Использование криптовалют.
                    Определение субъектов финансового мониторинга:
                    
                    Укажи, через каких субъектов финансового мониторинга (например, БВУ, МФО) совершались операции, в соответствии со статьей 3 Закона о ПОД/ФТ.
                    Генерация детализированных описаний:
                    
                    Создай отчеты, которые детализируют схемы отмывания денег, включая методы, суммы и задействованные субъекты.
                    Обработка текстов:
                    
                    Проанализируй тексты на русском и казахском языках, извлеки данные, найди ключевые статьи УК РК и опиши их содержание.
                    Создание визуализированных отчетов:
                    
                    Для раздела «Уязвимости»:
                    Создай таблицы и диаграммы по видам проверок.
                    Отобрази данные по субъектам финансового мониторинга.
                    Определи уязвимые направления в финансовом мониторинге.
                    Для раздела «Угрозы»:
                    Составь статистику по количеству дел, связанных с статьей 218 УК РК.
                    Классифицируй предикатные преступления.
                    Опиши схемы отмывания денег, указав суммы, методы и субъекты.
                    Используй данные из предоставленного списка дел: {filtered_results}. Если данных недостаточно, проведи анализ без их использования.
                    Оставляй ссылки на источники откуда ты брал информацию.
            """,
        },
    ]

    response = openai_client.chat.completions.create(
        model="o1-preview",
        messages=messages,
    )
    content = response.choices[0].message.content

    return content

# Интерфейс Streamlit
st.title("Система анализа текстов и файлов для классификации уголовных дел")

# Ввод текста
text_input = st.text_area("Введите текст для анализа")

# Загрузка файла
uploaded_file = st.file_uploader("Или загрузите файл (.docx или .pdf)", type=["docx", "pdf"])

if st.button("Анализировать"):
    if text_input:
        text_to_analyze = text_input
    elif uploaded_file:
        text_to_analyze = extract_text_from_file(uploaded_file)
    else:
        st.error("Введите текст или загрузите файл для анализа.")
        st.stop()

    with st.spinner("Обработка данных..."):
        result = get_similar_cases(text_to_analyze)
    st.success("Анализ завершен!")
    st.write("Результат:")
    st.write(result)
