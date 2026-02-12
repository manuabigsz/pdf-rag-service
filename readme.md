## PDF RAG Service

Sistema de perguntas e respostas sobre PDFs usando Retrieval-Augmented Generation (RAG), com suporte a **OpenAI** e **Ollama** como provedores de LLM e embeddings.

O usuário faz upload de documentos PDF, o texto é extraído, fatiado em chunks, indexado em um vetor store (**Chroma** via LangChain) e depois perguntas são respondidas com base nesses documentos.

---

### Arquitetura

- **API**: `FastAPI` em `services/api`
  - `POST /documents`: upload e indexação de PDFs
  - `POST /question`: responde perguntas usando RAG (LLM + Chroma)
  - `GET /health`: checagem de saúde simples
- **Vector Store**: `Chroma` persistente em disco (`./data/chroma_*`)
- **LLMs / Embeddings**:
  - **OpenAI** (`gpt-4o-mini`, `text-embedding-3-small` por padrão)
  - **Ollama** (`llama3.2:3b` + `nomic-embed-text` por padrão)
- **UI opcional**: `Streamlit` em `services/ui/streamlit_app.py` para upload e perguntas via navegador.

---

### Requisitos

- **Python 3.12+**
- **Poetry ou pip/venv** (a depender do seu fluxo; o Dockerfile usa `pip` + `requirements.txt`)
- **Docker** (opcional, mas recomendado para rodar a API)
- Conta e chave de API da **OpenAI** (se usar OpenAI)
- **Ollama** instalado e rodando localmente (se usar Ollama)

---

### Variáveis de ambiente

As principais variáveis estão centralizadas em `services/api/libs/utils/envs.py`:

- **Seleção de provedores**
  - `LLM_PROVIDER` (default: `openai`)  
    - Valores possíveis: `openai`, `ollama`
  - `EMBEDDING_PROVIDER` (default: `openai`)  
    - Valores possíveis: `openai`, `ollama`

- **OpenAI**
  - `OPENAI_API_KEY` **(obrigatório se usar OpenAI)**
  - `OPENAI_LLM_MODEL` (default: `gpt-4o-mini`)
  - `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)

- **Ollama**
  - `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
  - `OLLAMA_LLM_MODEL` (default: `llama3.2:3b`)
  - `OLLAMA_EMBEDDING_MODEL` (default: `nomic-embed-text`)

- **Chroma / RAG**
  - `CHROMA_PERSIST_DIR` (default:  
    - `./data/chroma_openai` quando `EMBEDDING_PROVIDER=openai`  
    - `./data/chroma_ollama` quando `EMBEDDING_PROVIDER=ollama`
  - `CHROMA_COLLECTION` (default: `documents`)
  - `CHUNK_SIZE` (default: `1000`)
  - `CHUNK_OVERLAP` (default: `150`)
  - `TOP_K` (default: `10` documentos mais similares por pergunta)

- **UI**
  - `API_BASE_URL` (default: `http://localhost:8000`) – usado pelo Streamlit.

Sugestão de `.env`:

```bash
OPENAI_API_KEY=xxxxx

LLM_PROVIDER=ollama ou openai
EMBEDDING_PROVIDER=ollama ou openai

OLLAMA_BASE_URL=http://host.docker.internal:11434  # se rodando em Docker
OLLAMA_LLM_MODEL=llama3.2:3b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

TOP_K=10
```

---

### Como rodar com Docker Compose (recomendado)

O `docker-compose.yml` na raiz sobe **backend (API)** e **frontend (UI Streamlit)** juntos.

- Serviço **api**  
  - Build em `./services/api` (usa `services/api/Dockerfile`)  
  - Porta exposta: `8000` → `http://localhost:8000`  
  - Carrega variáveis de `envs/api.dev.env`  
  - Usa `uvicorn main:app --reload`

- Serviço **ui**  
  - Build em `./services/ui`  
  - Porta exposta: `8501` → `http://localhost:8501`  
  - Usa `API_BASE_URL=http://api:8000` para falar com o backend no container

Passos:

1. **Configurar variáveis da API**

   Edite `envs/api.dev.env` conforme o provider desejado (OpenAI ou Ollama), seguindo a seção de variáveis de ambiente deste README.

2. **Subir todo o stack (API + UI)**

   Na raiz do projeto (`pdf-rag-service`):

   ```bash
   docker-compose up --build
   ```

   - Backend: `http://localhost:8000`
   - Frontend (Streamlit): `http://localhost:8501`

3. **Parar os serviços**

   ```bash
   docker-compose down
   ```

---

### Como rodar a UI (Streamlit)

A UI é opcional, mas facilita o teste:

```bash
cd services/ui
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # ou inclua streamlit/requests em seu env
```

Garanta que a API está rodando em `http://localhost:8000` ou ajuste `API_BASE_URL`.

Depois:

```bash
streamlit run streamlit_app.py
```

Acesse o endereço mostrado no terminal (por padrão `http://localhost:8501`).

---

### Endpoints da API

#### `POST /documents`

Faz upload de um ou mais PDFs e indexa os textos no Chroma.

- **Content-Type**: `multipart/form-data`
- Campo: `files` (um ou mais arquivos PDF)

Exemplo com `curl`:

```bash
curl -X POST http://localhost:8000/documents/ \
  -F "files=@manual1.pdf" \
  -F "files=@manual2.pdf"
```

Resposta esperada:

```json
{
  "message": "Documents processed successfully",
  "documents_indexed": 2,
  "total_chunks": 128
}
```

#### `POST /question`

Faz uma pergunta com base nos PDFs já indexados.

- **Content-Type**: `application/json`

Body:

```json
{
  "question": "What is the power consumption of the motor?"
}
```

Exemplo com `curl`:

```bash
curl -X POST http://localhost:8000/question/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the power consumption of the motor?"}'
```

Resposta (exemplo):

```json
{
  "answer": "The motor's power consumption is 2.3 kW.",
  "references": [
    "the motor xxx has requires 2.3kw to operate at a 60hz line frequency"
  ]
}
```

#### `GET /health`

Endpoint simples para ver se a API está de pé:

```bash
curl http://localhost:8000/health/
```

Resposta:

```json
{ "status": "ok" }
```

---

### Fluxo de uso

1. Suba a API (local ou via Docker), configurando o provider desejado (**OpenAI** ou **Ollama**).
2. Envie PDFs para `/documents`.
3. Faça perguntas para `/question`.
4. Opcionalmente, use o Streamlit para uma experiência web completa.

---

### Critérios do desafio atendidos

- **Upload e extração de PDFs**: via `pdfplumber` em `pdf_service`, com limpeza de texto.
- **Chunking e embeddings**: `RecursiveCharacterTextSplitter` + embeddings configuráveis (OpenAI/Ollama) em `vector_service`.
- **Armazenamento e recuperação**: `Chroma` persistente em disco, com `retriever` baseado em similaridade.
- **LLM para respostas**: `ChatOpenAI` ou `ChatOllama`, com prompt estruturado e referências das fontes.
- **API clara**: endpoints `/documents`, `/question`, `/health` com tipagem via Pydantic.
- **Developer UX**: suporte a Docker, UI em Streamlit, configuração por variáveis de ambiente e este README com exemplos de requisição.

