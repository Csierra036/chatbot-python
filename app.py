import google.generativeai as genai
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma

from dotenv import load_dotenv
import os

RELATIVE_PATH = os.getcwd() + "/src/pdf/"
CHROMA_PERSIST_DIRECTORY = "chroma_db" # <--- AQUI LA RUTA DE PERSISTENCIA


def template_prompt_gemma(context: str, query_text: str):

    prompt_template = f"""
        Eres un asistente académico virtual de la universidad, especializado en ingeniería informática. Tu misión es ayudar a estudiantes, docentes e investigadores.
        La mayoría de tus usuarios son principiantes en los temas consultados, aunque algunos pueden tener conocimientos avanzados.

        **TAREA PRINCIPAL:**
        Debes responder la "Pregunta del Usuario" basándote ÚNICA Y EXCLUSIVAMENTE en la información contenida en el "Contexto Proporcionado". Este contexto proviene de la base de datos de trabajos de grado, tesis e investigaciones de la universidad. NO DEBES USAR NINGÚN CONOCIMIENTO EXTERNO NI ACCEDER A INTERNET.

        **INSTRUCCIONES DETALLADAS PARA LA RESPUESTA:**

        1.  **Idioma:** La respuesta debe ser siempre en **español**.
        2.  **Fuente de Información:** Limita tu respuesta estrictamente al "Contexto Proporcionado". Si la información necesaria para responder no se encuentra en el contexto, debes indicar amablemente que la información no está disponible en los documentos consultados.
        3.  **Contenido General:** Extrae y presenta conceptos académicos relevantes, definiciones, fundamentos teóricos y/o aplicaciones básicas que se encuentren explícitamente en el "Contexto Proporcionado" y que respondan a la "Pregunta del Usuario".
        4.  **Prioridad en Contenido:** Dentro del contexto, da preferencia a conceptos actualizados y enfoques recientes si la información lo permite.
        5.  **Formato y Detalle Específico del Contenido:**
            * **Si la "Pregunta del Usuario" se refiere a elementos listados (con viñetas o numeración) en el "Contexto Proporcionado", y esos elementos tienen definiciones o descripciones asociadas DENTRO de dicho contexto:**
                * **Para CADA elemento de la lista, debes presentar primero el nombre del elemento y LUEGO su definición o descripción tal como aparece en el "Contexto Proporcionado".** Intenta ser lo más fiel posible al texto original del contexto para estas definiciones/descripciones.
                * Si después de citar la definición/descripción del contexto consideras que una breve explicación adicional en un párrafo aparte puede ayudar a la comprensión, puedes añadirla, siempre y cuando el conjunto se mantenga claro y dentro del límite de caracteres.
                * El objetivo es que el usuario reciba tanto el nombre del elemento como su explicación/definición directamente del contexto.
            * **Si la "Pregunta del Usuario" es más general, ambigua, abstracta o no se refiere a una lista específica con detalles explícitos en el contexto:** Ofrece una explicación breve, clara y concisa sobre el tema general solicitado, basándote siempre en la información disponible en el "Contexto Proporcionado".
        6.  **Extensión Total:** La respuesta completa (incluyendo todos los elementos de una lista y sus descripciones/explicaciones, si aplica) NO debe exceder los **700 caracteres**. Sé breve y ve al grano. Si la información completa de una lista con todas sus descripciones es demasiado extensa para este límite, prioriza explicar los primeros elementos de manera completa o resume concisamente la descripción de cada uno, basándote fielmente en el texto del contexto.
        7.  **Nivel de Detalle General:** La explicación debe ser clara y permitir la comprensión del tema incluso sin conocimientos previos profundos por parte del usuario.
        8.  **Tono:** Utiliza un tono **amigable, claro y objetivo**.
        9.  **Estilo y Claridad:**
            * Usa un lenguaje **accesible para estudiantes**.
            * Evita tecnicismos complejos. Si un tecnicismo es esencial y está presente en el contexto, explícalo brevemente.
            * **Evita definiciones circulares**.

        ---
        **Contexto Proporcionado:**
        {context}
        ---
        **Pregunta del Usuario:**
        {query_text}
        ---
        **Respuesta (basada únicamente en el contexto, siguiendo estrictamente las instrucciones de formato y detalle, máximo 700 caracteres, en español, amigable y clara):**
    """
    return prompt_template

def load_document():
    # Asegúrate de que el directorio exista
    if not os.path.isdir(RELATIVE_PATH):
        print(f"Error: El directorio '{RELATIVE_PATH}' no existe o no es un directorio válido.")
        return [] # Devuelve una lista vacía si el directorio no es válido

    document_loader = PyPDFDirectoryLoader(RELATIVE_PATH)
    return document_loader.load()

def split_document(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function=len,
        separators=["\n\n", "\n"]
    )
    return text_splitter.split_documents(documents)

def get_embedding_document_function():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key = os.getenv('GOOGLE_API_KEY'),
        task_type="RETRIEVAL_DOCUMENT"
    )
    return embeddings

def get_embedding_query_function():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key = os.getenv('GOOGLE_API_KEY'),
        task_type="RETRIEVAL_QUERY"
    )
    return embeddings

def add_to_chroma(chunks: list[Document], batch_size: int = 100):
    db = Chroma(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=get_embedding_document_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    print(f"👉 Adding new documents: {len(new_chunks)}")

    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i+batch_size]
        batch_ids = [chunk.metadata["id"] for chunk in batch]
        try:
            db.add_documents(batch, ids=batch_ids)
        except Exception as e:
            print(f"Error al agregar el batch {i}-{i+batch_size}: {e}")

def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def query_rag(query_text: str):
    # Carga la base de datos existente desde la ruta de persistencia
    db = Chroma(
        persist_directory=CHROMA_PERSIST_DIRECTORY, embedding_function=get_embedding_query_function()
    )
    retriever = db.as_retriever(search_kwargs={'k': 3})
    docs = retriever.get_relevant_documents(query_text)
    context = "\n".join([doc.page_content for doc in docs])

    try:
        response = genai.GenerativeModel("gemma-3-27b-it").generate_content(
            template_prompt_gemma(context, query_text)
        )
        print("Respuesta:", response.text)
    except Exception as e:
        print(f"Error al generar la respuesta con Gemma: {e}")
        # Aquí podrías tener una lógica de fallback o un mensaje de error amigable

def main():
    load_dotenv()
    documents = load_document()
    if documents: # Solo procesa si hay documentos cargados
        chunks = split_document(documents)
        print(f"Número de chunks: {len(chunks)}")
        # print(chunks[0]) # Solo para depuración
        add_to_chroma(chunks)
        # Ejemplo de consulta
        query_rag("¿Cuales son las tecnicas de comunicacion E/S?")
    else:
        print("No se encontraron documentos para procesar.")

if __name__ == "__main__":
    main()