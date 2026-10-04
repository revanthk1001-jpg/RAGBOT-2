import os, re, json, time, uuid, tempfile, threading
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS

from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from langfuse.langchain import CallbackHandler

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "150"))  # biggest latency lever
TOP_K = int(os.getenv("TOP_K", "3"))
JUDGE_MAX_NEW_TOKENS = 600
INDEX_DIR = "faiss-index"

_embeddings = None
_llm = None
_db = None
_gen_lock = threading.Lock()  # one generation at a time; stops eval jobs fighting user queries
EVAL_RESULTS: dict = {}


class QueryRequest(BaseModel):
    query: str
    evaluate: bool = False  # runs in background; fetch via GET /eval/{id}


# ---------- model loading ----------
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings


def get_llm():
    global _llm
    if _llm is None:
        cuda = torch.cuda.is_available()
        _llm = HuggingFacePipeline.from_model_id(
            model_id=LLM_MODEL,
            task="text-generation",
            device=0 if cuda else -1,
            # fp32 is usually faster than bf16 on CPUs without bf16 hardware support
            model_kwargs={"torch_dtype": torch.float16 if cuda else torch.float32},
            pipeline_kwargs=dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, return_full_text=False),
        )
    return _llm


def get_db(required=True):
    global _db
    if _db is None:
        if not os.path.exists(INDEX_DIR):
            if required:
                raise RuntimeError("Please ingest a PDF first.")
            return None
        _db = FAISS.load_local(INDEX_DIR, get_embeddings(), allow_dangerous_deserialization=True)
    return _db


# ---------- generation helpers ----------
def fmt(user_text: str) -> str:
    tok = get_llm().pipeline.tokenizer
    return tok.apply_chat_template(
        [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True
    )


def generate(user_text: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    langfuse_handler = CallbackHandler()
    with _gen_lock:
        return get_llm().invoke(
            fmt(user_text),
            config={"callbacks": [langfuse_handler]},
            pipeline_kwargs={"max_new_tokens": max_new_tokens, "do_sample": False},
        ).strip()


def stream_generate(user_text: str):
    langfuse_handler = CallbackHandler()
    with _gen_lock:
        for tok in get_llm().stream(
            fmt(user_text),
            config={"callbacks": [langfuse_handler]},
            pipeline_kwargs={"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False},
        ):
            yield tok


def build_prompt(context: list, query: str) -> str:
    return (
        "Answer the question using ONLY the context below. Be concise. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Context:\n" + "\n\n".join(context) + f"\n\nQuestion: {query}\nAnswer:"
    )


def retrieve(query: str):
    docs = get_db().similarity_search(query, k=TOP_K)
    context = [d.page_content for d in docs]
    sources = sorted({f"{d.metadata.get('source')} (p.{d.metadata.get('page', 0) + 1})" for d in docs})
    return context, sources


# ---------- DeepEval judge ----------
class JudgeLLM(DeepEvalBaseLLM):
    def load_model(self):
        return get_llm()

    def _parse(self, text: str, schema):
        if schema is None:
            return text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Judge returned no JSON: {text[:200]}")
        return schema(**json.loads(match.group()))

    def generate(self, prompt: str, schema: Optional[BaseModel] = None):
        return self._parse(generate(prompt, JUDGE_MAX_NEW_TOKENS), schema)

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None):
        return await run_in_threadpool(self.generate, prompt, schema)

    def get_model_name(self):
        return LLM_MODEL


def evaluate_answer_sync(query: str, context: list, answer: str) -> dict:
    test_case = LLMTestCase(input=query, actual_output=answer, retrieval_context=context)
    judge = JudgeLLM()
    out = {}
    for name, metric in (
        ("faithfulness", FaithfulnessMetric(model=judge, async_mode=False)),
        ("answer_relevancy", AnswerRelevancyMetric(model=judge, async_mode=False)),
    ):
        try:
            metric.measure(test_case)
            out[name] = metric.score
        except Exception as e:  # small judges often fail JSON parsing
            out[name] = None
            out[f"{name}_error"] = str(e)[:200]
    return out


def eval_job(eval_id: str, query: str, context: list, answer: str):
    t = time.time()
    res = evaluate_answer_sync(query, context, answer)
    EVAL_RESULTS[eval_id] = {"status": "done", "eval_seconds": round(time.time() - t, 1), **res}


# ---------- app ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # load + warm up once so the first user request isn't slow
    get_embeddings()
    get_llm()
    get_db(required=False)
    generate("Say hi.", max_new_tokens=2)
    yield


app = FastAPI(title="Enterprise RAG API (local HF models)", lifespan=lifespan)


def ingest_pdf(file_bytes: bytes, filename: str) -> int:
    global _db
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(file_bytes)
        temp_path = f.name
    try:
        docs = PyPDFLoader(temp_path).load()
    finally:
        os.remove(temp_path)

    safe_name = os.path.basename(filename)
    for d in docs:
        d.metadata["source"] = safe_name

    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50).split_documents(docs)
    store = FAISS.from_documents(chunks, get_embeddings())
    store.save_local(INDEX_DIR)
    _db = store
    return len(chunks)


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    data = await file.read()
    chunks = await run_in_threadpool(ingest_pdf, data, file.filename)
    return {"status": "success", "chunks_embedded": chunks}


@app.post("/query")
def query(request: QueryRequest, background: BackgroundTasks):
    try:
        t0 = time.time()
        context, sources = retrieve(request.query)
        t_retrieve = time.time() - t0
        answer = generate(build_prompt(context, request.query))
        result = {
            "answer": answer,
            "sources": sources,
            "retrieval_seconds": round(t_retrieve, 2),
            "latency": round(time.time() - t0, 2),
        }
        if request.evaluate:
            eval_id = uuid.uuid4().hex
            EVAL_RESULTS[eval_id] = {"status": "running"}
            background.add_task(eval_job, eval_id, request.query, context, answer)
            result["eval_id"] = eval_id
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
def query_stream(request: QueryRequest):
    try:
        context, sources = retrieve(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return StreamingResponse(
        stream_generate(build_prompt(context, request.query)),
        media_type="text/plain",
        headers={"X-Sources": json.dumps(sources)},  # ascii-escaped, header-safe
    )


@app.get("/eval/{eval_id}")
def get_eval(eval_id: str):
    if eval_id not in EVAL_RESULTS:
        raise HTTPException(status_code=404, detail="Unknown eval id")
    return EVAL_RESULTS[eval_id]