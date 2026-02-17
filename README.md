# 📚 Assistant Personnel de Lecture de Documents (PDF) — Streamlit + LangChain (RAG)

Cette application Streamlit permet d’importer **un ou plusieurs fichiers PDF**, de créer une **base vectorielle FAISS** à partir des contenus, puis de poser des questions.  
Les réponses sont générées par un modèle OpenAI via **LangChain**, en se basant **uniquement** sur le contenu des PDFs (RAG).

---

## ✅ Fonctionnalités

- Import de **plusieurs PDFs** (upload Streamlit)
- Découpage du texte en chunks (RecursiveCharacterTextSplitter)
- Création d’**embeddings** (`text-embedding-3-large`)
- Indexation dans **FAISS**
- Recherche sémantique + génération de réponse (chaîne RAG)
- Gestion d’erreurs avec affichage du traceback

---

## 🧱 Stack / Librairies

- [Streamlit](https://streamlit.io/)
- LangChain
- `langchain_openai` (ChatOpenAI + OpenAIEmbeddings)
- FAISS (vector store)
- PyPDFLoader
- python-dotenv

---

## 📁 Structure de projet (exemple)

mon-projet/

app.py

requirements.txt

../.env # (selon ton load_dotenv("../.env"))

## 🔐 Variables d’environnement

Crée un fichier `.env` contenant ta clé OpenAI :


exemple: OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx


---

## 🛠️ Installation

### 1) Créer un environnement virtuel 

python -m venv .venv
.venv\Scripts\activate

puis lancez la commande suivante pour installer les dependances


pip install -r requirements.txt

### Demarrez l'application avec la commande streamlit run app.py


## Utilisation

1. Ouvrez l'app

2. Importer un ou plusieurs fichiers PDF

3. Attends la création des plongements / index FAISS

4. Poser une question dans le champ prévu

5. Lis la réponse générée (basée sur les documents)




