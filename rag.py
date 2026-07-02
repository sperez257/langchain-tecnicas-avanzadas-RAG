import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from transformers import AutoTokenizer
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

pdfs = DirectoryLoader("./documents", glob="*.pdf").load()
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=1250,
    chunk_overlap=150
)

fragmentos = splitter.split_documents(pdfs)

embeddings = OllamaEmbeddings(model="bge-m3:567m")

# print(fragmentos[0])

prompt = ChatPromptTemplate(
    [
        ("system","Responde utilizando exclusivamente el contenido que se anexa a continuación: \nContexto:\n{contexto}"),
        ("human", "{query}")
    ]
)

vector_store = FAISS.from_documents(documents=fragmentos, embedding=embeddings)
retriver = vector_store.as_retriever()

model = OllamaLLM(model='gemma3:4b')

cadena = prompt | model | StrOutputParser()

pregunta = 'Como solicitar el seguro del viaje?'

# chunks = retriver.invoke(pregunta)
# contexto = "\n\n".join(chunk.page_content for chunk in chunks)
# cadena.invoke({'query':pregunta, 'contexto':contexto})



query_model = OllamaLLM(model="gemma3:4b")

rewriter_prompt_template = """
Genera la consulta de búsqueda para la base de datos de vectores (Vector DB) a partir de una pregunta del usuario,
permitiendo una respuesta más precisa por medio de la búsqueda semántica.
Basta devolver la consulta revisada del Vector DB, entre comillas.

# PREGUNTA DEL USUARIO: {user_question}
# CONSULTA REVISADA DEL VECTOR DB:
"""

rewriter_prompt = PromptTemplate.from_template(rewriter_prompt_template)
rewriter_chain = rewriter_prompt | query_model | StrOutputParser()

# rag_chain = (
#     {
#         "contexto": RunnablePassthrough() | rewriter_chain | retriver,
#         "query": RunnablePassthrough()
#     } | prompt | model | StrOutputParser()
# )

# rag_chain.invoke(pregunta)

template_multipregunta = """
Eres un asistente de modelo de lenguajes de IA. Tu tarea es generar cinco versiones diferentes de la pregunta 
del usuario para recuperar documentos relevantes de una base de datos vectorial. Al generar multiples
perspectivas sobre la pregunta del usuario, tu objetivo es auxiliar al usuario a superar algunas de las
limitaciones de la búsqueda por similitud basada en distancia. Debes generar únicamente las preguntas alternativas
separadas en filas diferentes (new line) sin ningún texto adicional.

# PREGUNTA ORIGINAL: {question}

# FORMATO DE SALIDA :
["primera pregunta","segunda pregunta",...,"quinta pregunta"]
"""

prompt_multipregunta = PromptTemplate.from_template(template_multipregunta)
chain_multipregunta = prompt_multipregunta | model | CommaSeparatedListOutputParser()

preguntas = chain_multipregunta.invoke(pregunta)
print(preguntas)

rag_chain = (
    {
        "contexto": RunnablePassthrough() | retriver,
        "query": RunnablePassthrough()
    } | prompt | model | StrOutputParser()
)

[rag_chain.invoke(p) for p in preguntas]