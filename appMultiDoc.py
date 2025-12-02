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

# -------------------------
# Initialisation
# -------------------------
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template("""
<context>
{context}
</context>

Question: {input}
""")


# -------------------------
# Fonctions
# -------------------------

def load_documents(uploaded_files):
    """Charge et découpe plusieurs PDF."""
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    for idx, uploaded_file in enumerate(uploaded_files):
        temp_path = f"temp_{idx}.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        split_docs = text_splitter.split_documents(docs)
        all_docs.extend(split_docs)

    return all_docs


def load_embeddings(docs):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    db = FAISS.from_documents(docs, embeddings)
    return db.as_retriever()


def query(retriever, question):
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)
    return chain.invoke({"input": question})


# -------------------------
# Interface type ChatGPT
# -------------------------

st.set_page_config(page_title="Chat PDF", layout="wide")

# CSS pour fixer la barre en bas
st.markdown("""
    <style>
        .chat-input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            padding: 10px;
            background: white;
            box-shadow: 0 -3px 6px rgba(0,0,0,0.1);
            z-index: 9999;
        }

        .chat-history {
            padding-bottom: 140px; /* espace réservé pour la barre fixe */
        }

        textarea {
            font-size: 18px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📚 Chat avec plusieurs PDF – Interface type ChatGPT")

uploaded_files = st.file_uploader("Dépose tes PDF", type=["pdf"], accept_multiple_files=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None


# --- Chargement des documents ---
if uploaded_files and st.session_state.retriever is None:
    with st.spinner("Analyse des documents..."):
        docs = load_documents(uploaded_files)
        st.session_state.retriever = load_embeddings(docs)
    st.success("Documents indexés ✔️")


# -------------------------
# AFFICHAGE DES MESSAGES
# -------------------------
st.markdown('<div class="chat-history">', unsafe_allow_html=True)

for role, msg in st.session_state.messages:
    if role == "user":
        st.markdown(f"🧑 **Vous :** {msg}")
    else:
        st.markdown(f"🤖 **Assistant :** {msg}")

st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# BARRE DE SAISIE FIXÉE EN BAS
# -------------------------
with st.container():
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)

    with st.form("chat-form", clear_on_submit=True):
        question = st.text_area("Votre message :", height=60, label_visibility="collapsed")
        submitted = st.form_submit_button("Envoyer")

        if submitted and question:
            # Ajouter message utilisateur
            st.session_state.messages.append(("user", question))

            # Obtenir réponse
            if st.session_state.retriever:
                with st.spinner("Réflexion..."):
                    try:
                        response = query(st.session_state.retriever, question)
                        answer = response["answer"]
                    except Exception as e:
                        answer = "Erreur :\n" + traceback.format_exc()
            else:
                answer = "⚠️ Aucun document chargé."

            # Ajouter réponse IA
            st.session_state.messages.append(("assistant", answer))

            # Refresh
            st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)
