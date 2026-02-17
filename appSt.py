import streamlit as st
import traceback
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os


# Initialisation de notre clé API openAI et chhargement du modele de language gpt

load_dotenv("../.env")

llm = ChatOpenAI(model="gpt-4.1")

prompt = ChatPromptTemplate.from_template("""
Tu es un assistant spécialisé dans l'analyse de documents.
Répond seulement à partir du contexte fourni. 
Si la réponse n’est pas dans les documents, dis que tu ne sais pas.
<context>
{context}
</context>

Question: {input}
reponse:
""")


# chargement des documents

def load_documents(uploaded_files):
    """ Charge et découpe plusieurs PDF importés dans Streamlit """

    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0)

    for idx, uploaded_file in enumerate(uploaded_files):
        # Sauvegarde temporaire du PDF
        temp_path = f"temp_{idx}.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Charge PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Découpe
        split_docs = text_splitter.split_documents(docs)
        all_docs.extend(split_docs)

        os.remove(temp_path)

    return all_docs

# decoupage des document et creation des vecteurs numeriques ou embeddings


def load_embeddings(docs):
    """ Crée la base vectorielle FAISS pour TOUS les documents """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    db = FAISS.from_documents(docs, embeddings)
    return db.as_retriever()

# chercher la reponse d'une question


def query(retriever, question):
    """ Exécute la chaîne de RAG """
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)
    return chain.invoke({"input": question})


# Interface utilisateur pour charger des documents et interagir avec l'assistant
st.title("📚 Assistant Personnel pour lecture de documents")

st.write("Importe **plusieurs fichiers PDF** et pose des questions sur tous les documents combinés !")

uploaded_files = st.file_uploader("Dépose un ou plusieurs PDF", type=[
                                  "pdf"], accept_multiple_files=True)


# Stockage dans session_state
if "docs" not in st.session_state:
    st.session_state.docs = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None


if uploaded_files:
    st.success(f"{len(uploaded_files)} documents importés ✔️")

    if st.session_state.docs is None:
        with st.spinner("Lecture et découpage des documents…"):
            st.session_state.docs = load_documents(uploaded_files)

    if st.session_state.retriever is None:
        with st.spinner("Création des embeddings…"):
            st.session_state.retriever = load_embeddings(st.session_state.docs)

    st.success("Base vectorielle prête ✔️")

    question = st.text_input("Pose ta question ici 👇")

    if question:
        try:
            with st.spinner("Analyse des documents…"):
                response = query(st.session_state.retriever, question)

            st.subheader("🧠 Réponse :")
            st.write(response["answer"])

        except Exception:
            st.error("Une erreur est survenue.")
            st.code(traceback.format_exc())

else:
    st.info("Importe un ou plusieurs PDF pour commencer…")
