import streamlit as st
import requests
import json

API_URL = "http://127.0.0.1:8001"

st.set_page_config(page_title="Enterprise RAG Chat", layout="centered")
st.title("💬 Enterprise RAG Chat")

# 1. Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Sidebar for PDF Ingestion (keeps chat area clean)
with st.sidebar:
    st.header("📄 Knowledge Base")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file and st.button("Ingest PDF"):
        with st.spinner("Processing & Embedding..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            res = requests.post(f"{API_URL}/ingest", files=files)
            if res.status_code == 200:
                st.success(f"Success! Embedded {res.json()['chunks_embedded']} chunks.")
            else:
                st.error(res.text)
                
    st.divider()
    run_eval = st.checkbox("Evaluate Answer (Slow)", value=False, help="Run DeepEval metrics. This is very slow on local hardware.")

# 3. Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show metrics cleanly in an expander for assistant messages
        if msg["role"] == "assistant" and "metrics" in msg:
            with st.expander("📊 View Sources & Evaluation Metrics"):
                st.write(f"**Sources:** {', '.join(msg['metrics'].get('sources', [])) if msg['metrics'].get('sources') else 'None'}")
                
                eval_id = msg['metrics'].get('eval_id')
                if eval_id:
                    eval_res = requests.get(f"{API_URL}/eval/{eval_id}")
                    if eval_res.status_code == 200:
                        eval_data = eval_res.json()
                        if eval_data.get("status") == "running":
                            st.write("*Evaluation is currently running in the background...*")
                        else:
                            f_score = eval_data.get('faithfulness')
                            r_score = eval_data.get('answer_relevancy')
                            if f_score is not None:
                                st.write(f"**Faithfulness:** {f_score:.2f}")
                            if r_score is not None:
                                st.write(f"**Relevancy:** {r_score:.2f}")
                            if eval_data.get('faithfulness_error'):
                                st.error(f"Faithfulness Error: {eval_data['faithfulness_error']}")
                    else:
                        st.write("*Failed to fetch evaluation results.*")
                else:
                    st.write("*Evaluation was skipped for this query.*")

# 4. Chat input at the bottom
if prompt := st.chat_input("Ask a question about your document..."):
    # Add and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        if run_eval:
            with st.spinner("Thinking (and evaluating in background)..."):
                res = requests.post(f"{API_URL}/query", json={"query": prompt, "evaluate": True})
                if res.status_code == 200:
                    data = res.json()
                    st.markdown(data["answer"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "metrics": {
                            "sources": data.get("sources", []),
                            "eval_id": data.get("eval_id")
                        }
                    })
                else:
                    st.error(f"API Error: {res.text}")
        else:
            with requests.post(f"{API_URL}/query/stream", json={"query": prompt}, stream=True) as res:
                if res.status_code == 200:
                    sources = json.loads(res.headers.get("X-Sources", "[]"))
                    placeholder = st.empty()
                    full_response = ""
                    for chunk in res.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            full_response += chunk
                            placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "metrics": {
                            "sources": sources
                        }
                    })
                else:
                    st.error(f"API Error: {res.text}")


