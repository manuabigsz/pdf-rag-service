## PDF RAG Service

A question-and-answer system over PDFs using **Retrieval-Augmented Generation (RAG)**, with support for **OpenAI** and **Ollama** as LLM and embedding providers.

Users upload PDF documents, the text is extracted, split into chunks, indexed in a vector store (**Chroma** via LangChain), and questions are answered based on the indexed content.

---

### Architecture

* **API**: `FastAPI` in `services/api`

  * `POST /documents`: upload and index PDFs
  * `POST /question`: answer questions using RAG (LLM + Chroma)
  * `GET /health`: simple health check
* **Vector Store**: Persistent **Chroma** stored on disk (`./data/chroma_*`)
* **LLMs / Embeddings**:

  * **OpenAI** (`gpt-4o-mini`, `text-embedding-3-small` by default)
  * **Ollama** (`llama3.2:3b` + `nomic-embed-text` by default)
* **Optional UI**: `Streamlit` in `services/ui/streamlit_app.py` for browser-based uploads and questions.

---

### Requirements

* **Python 3.12+**
* **Poetry or pip/venv** (depending on your workflow; the Dockerfile uses `pip` + `requirements.txt`)
* **Docker** (optional, but recommended to run the API)
* **OpenAI API key** (if using OpenAI)
* **Ollama** installed and running locally (if using Ollama)

---

### Environment Variables

Main environment variables are centralized in `services/api/libs/utils/envs.py`:

#### Provider Selection

* `LLM_PROVIDER` (default: `openai`)

  * Possible values: `openai`, `ollama`
* `EMBEDDING_PROVIDER` (default: `openai`)

  * Possible values: `openai`, `ollama`

#### OpenAI

* `OPENAI_API_KEY` **(required if using OpenAI)**
* `OPENAI_LLM_MODEL` (default: `gpt-4o-mini`)
* `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)

#### Ollama

* `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
* `OLLAMA_LLM_MODEL` (default: `llama3.2:3b`)
* `OLLAMA_EMBEDDING_MODEL` (default: `nomic-embed-text`)

#### Chroma / RAG

* `CHROMA_PERSIST_DIR` (default:

  * `./data/chroma_openai` when `EMBEDDING_PROVIDER=openai`
  * `./data/chroma_ollama` when `EMBEDDING_PROVIDER=ollama`)
* `CHROMA_COLLECTION` (default: `documents`)
* **`CHUNK_SIZE`** (default: `1000`) – Maximum size, in characters, of each chunk the PDF text is split into before generating embeddings. Smaller chunks tend to produce more precise answers for specific passages; larger chunks preserve more context. Adjust based on document type (e.g., 500–800 for technical manuals, 1200–1500 for long-form text).
* **`CHUNK_OVERLAP`** (default: `150`) – Number of overlapping characters between consecutive chunks to avoid cutting sentences in the middle and improve retrieval continuity.
* **`TOP_K`** (default: `10`) – Number of most similar chunks retrieved and sent to the LLM to generate the final answer.

#### UI

* `API_BASE_URL` (default: `http://localhost:8000`) – Used by Streamlit.

Example `.env`:

```bash
OPENAI_API_KEY=xxxxx

LLM_PROVIDER=ollama or openai
EMBEDDING_PROVIDER=ollama or openai

OLLAMA_BASE_URL=http://host.docker.internal:11434  # if running inside Docker
OLLAMA_LLM_MODEL=llama3.2:3b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

TOP_K=10
```

---

### Running with Ollama (Local Models)

To use **Ollama** as both LLM and embedding provider (without relying on OpenAI):

#### 1. Install Ollama

* Visit [https://ollama.com](https://ollama.com) and download the installer for your OS.
* Install and keep Ollama running (on Windows/Mac it runs in the background; on Linux use `ollama serve`).

#### 2. Download the Models

Run in your terminal:

```bash
# Language model (for answers)
ollama pull llama3.2:3b

# Embedding model (for semantic search)
ollama pull nomic-embed-text
```

If you want to use different models, update `OLLAMA_LLM_MODEL` and `OLLAMA_EMBEDDING_MODEL` in your `.env`.

#### 3. Configure the Environment

In `envs/api.dev.env` (or your `.env`):

```bash
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2:3b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

#### 4. If Running the API Inside Docker

The container must reach Ollama running on your machine. Use:

```bash
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

(The `docker-compose` file already configures `host.docker.internal` for this.)

After that, start the stack with:

```bash
docker-compose up --build
```

The system will use Ollama for both embeddings and response generation.

---

### Running with Docker Compose (Recommended)

The `docker-compose.yml` at the project root starts both **backend (API)** and **frontend (Streamlit UI)**.

#### API Service

* Build context: `./services/api` (uses `services/api/Dockerfile`)
* Exposed port: `8000` → `http://localhost:8000`
* Loads variables from `envs/api.dev.env`
* Runs with: `uvicorn main:app --reload`

#### UI Service

* Build context: `./services/ui`
* Exposed port: `8501` → `http://localhost:8501`
* Uses `API_BASE_URL=http://api:8000` to communicate with the backend container

#### Steps

1. **Configure API variables**

   Copy `envs/api.dev.env.example` to `envs/api.dev.env` and fill in the required values (e.g., `OPENAI_API_KEY`). Adjust the provider (OpenAI or Ollama) as needed.

2. **Start the full stack (API + UI)**

   From the project root (`pdf-rag-service`):

   ```bash
   docker-compose up --build
   ```

   * Backend: [http://localhost:8000](http://localhost:8000)
   * Frontend (Streamlit): [http://localhost:8501](http://localhost:8501)

3. **Stop services**

   ```bash
   docker-compose down
   ```

---

### API Endpoints

#### `POST /documents`

Uploads one or more PDFs and indexes their text into Chroma.

* **Content-Type**: `multipart/form-data`
* Field: `files` (one or more PDF files)

Example using `curl`:

```bash
curl -X POST http://localhost:8000/documents/ \
  -F "files=@manual1.pdf" \
  -F "files=@manual2.pdf"
```

Expected response:

```json
{
  "message": "Documents processed successfully",
  "documents_indexed": 2,
  "total_chunks": 128
}
```

---

#### `POST /question`

Asks a question based on the already indexed PDFs.

* **Content-Type**: `application/json`

Body:

```json
{
  "question": "what do you know about AC/DC motor installation and maintenance?"
}
```

Example using `curl`:

```bash
curl -X POST http://localhost:8000/question/ \
  -H "Content-Type: application/json" \
  -d '{"question": "what do you know about AC/DC motor installation and maintenance?"}'
```

Example response:

```json
{
  "answer": "The motor's power consumption is 2.3 kW.",
  "references": [
    "the motor xxx requires 2.3kw to operate at a 60hz line frequency"
  ]
}
```

---

#### `GET /health`

Simple endpoint to verify the API is running:

```bash
curl http://localhost:8000/health/
```

Response:

```json
{ "status": "ok" }
```

---

### Usage Flow

1. Start the API (locally or via Docker), configuring your desired provider (**OpenAI** or **Ollama**).
2. Upload PDFs to `/documents`.
3. Send questions to `/question`.
4. Optionally, use the Streamlit UI for a full web-based experience.


---

## 👨‍💻 Expert

<p>
    <img 
      align=left 
      margin=10 
      width=80 
      src="https://avatars.githubusercontent.com/u/80135269?v=4"
    />
    <p>&nbsp&nbsp&nbspManuela Bertella Ossanes<br>
    &nbsp&nbsp&nbsp
    <a href="https://avatars.githubusercontent.com/u/80135269?v=4">
    GitHub</a>&nbsp;|&nbsp;
    <a href="https://www.linkedin.com/in/manuela-bertella-ossanes-690166204/">LinkedIn</a>
&nbsp;|&nbsp;
    <a href="https://www.instagram.com/manuossz/">
    Instagram</a>
&nbsp;|&nbsp;</p>
</p>
<br/><br/>
<p>

---