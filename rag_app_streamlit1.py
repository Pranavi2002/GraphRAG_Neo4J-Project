# hallucinates or uses memory to generate answer

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from config import driver, embeddings, llm

vectorstore = None
# embeddings = OpenAIEmbeddings()
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------
# 1️⃣ Load & split documents
# ---------------------------
def load_documents(folder_path="uploads"):
    docs = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue
        docs.extend(loader.load())
    print(f"✅ Loaded {len(docs)} documents")
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)

# ---------------------------
# 2️⃣ Build FAISS vectorstore
# ---------------------------
def build_vectorstore(chunks):
    global vectorstore
    print("🔧 Building vector index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# ---------------------------
# 3️⃣ RAG query
# ---------------------------
def rag_query(question, k=3, use_docs_only=True):
    global vectorstore
    if vectorstore is None:
        raise ValueError("Vectorstore not built yet!")

    vector_docs = vectorstore.similarity_search(question, k=k)
    vector_context = "\n".join([doc.page_content for doc in vector_docs])

    if use_docs_only and not vector_context.strip():
        return f"⚠️ No info about '{question}' in uploaded docs.\n💡 Uncheck 'Use only uploaded documents' for general knowledge."

    system_prompt = "You are a helpful assistant. " + \
        ("Answer using **only provided document context**." if use_docs_only else "Use docs + general knowledge if needed.")
    
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{vector_context}\n\nQuestion: {question}"}
    ])
    answer = response.content
    if use_docs_only:
        answer += "\n💡 Uncheck to include general knowledge."
    return answer

# ---------------------------
# 4️⃣ Delete all docs locally
# ---------------------------
def delete_all_docs():
    for f in os.listdir("uploads"):
        os.remove(os.path.join("uploads", f))
    global vectorstore
    vectorstore = None
    print("🗑️ All documents deleted and vectorstore cleared.")

# ---------------------------
# 5️⃣ Delete a specific document
# ---------------------------
def delete_doc(doc_name):
    path = os.path.join("uploads", doc_name)
    if os.path.exists(path):
        os.remove(path)
        print(f"🗑️ Document '{doc_name}' deleted.")
    global vectorstore
    vectorstore = None