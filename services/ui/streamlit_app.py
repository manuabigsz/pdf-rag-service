import streamlit as st
import requests
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

DOCUMENTS_ENDPOINT = f"{API_BASE_URL}/documents"
QUESTION_ENDPOINT = f"{API_BASE_URL}/question"

st.set_page_config(
    page_title="RAG - PDF Q&A",
    layout="centered"
)

st.title("RAG com PDFs + LLM")
st.markdown("Faça upload de documentos e depois faça perguntas sobre eles.")

st.sidebar.header("Upload de Documentos")

uploaded_files = st.sidebar.file_uploader(
    "Escolha arquivos PDF",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("Enviar documentos"):
    if not uploaded_files:
        st.sidebar.error("Selecione ao menos um arquivo PDF.")
    else:
        with st.spinner("Processando documentos..."):
            files = [("files", (f.name, f.read(), "application/pdf")) for f in uploaded_files]

            response = requests.post(
                DOCUMENTS_ENDPOINT,
                files=files
            )

        if response.status_code == 200:
            data = response.json()
            st.sidebar.success("Documentos enviados com sucesso! ✅")
            st.sidebar.json(data)
        else:
            st.sidebar.error(f"Erro ao enviar documentos: {response.text}")

st.header("Faça uma pergunta sobre os documentos")

question = st.text_input("Digite sua pergunta aqui:")

if st.button("Perguntar"):
    if not question.strip():
        st.error("Digite uma pergunta primeiro.")
    else:
        with st.spinner("Buscando resposta..."):
            response = requests.post(
                QUESTION_ENDPOINT,
                json={"question": question}
            )

        if response.status_code == 200:
            data = response.json()

            st.subheader("💬 Resposta")
            st.write(data.get("answer", "Sem resposta"))

            st.subheader("📚 Referências")
            references = data.get("references", [])

            if references:
                for i, ref in enumerate(references, start=1):
                    with st.expander(f"Referência {i}"):
                        st.write(ref)
            else:
                st.info("Nenhuma referência retornada.")
        else:
            st.error(f"Erro ao buscar resposta: {response.text}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Status da API**")

try:
    health = requests.get(f"{API_BASE_URL}/health")
    st.sidebar.success("✅")
except Exception as e:
    st.sidebar.error("❌")
