import asyncio
import os
from io import StringIO

import jsonlines
import streamlit as st
import yaml

from adaptive_thinking.core.model import APIModel

st.set_page_config(page_title="JSONL Viewer & Editor", layout="wide")

# Ensure default session-state keys exist
if "data_loaded" not in st.session_state:
    st.session_state["data_loaded"] = False

st.title("📄 JSONL QA Viewer & Editor")
st.markdown("## Upload and edit a JSONL file below 👇")

uploaded_file = st.sidebar.file_uploader("📤 Upload your .jsonl file", type=["jsonl"])

# Add "Start with empty file" button
if st.sidebar.button("🆕 Start with empty file"):
    st.session_state.data = [
        {
            "index": 0,
            "question": "Evaluate the integral \\( \\int_0^1 (3x^2 + 2x) \\, dx \\).",
            "answer": "\\( \\boxed{2} \\)",
            "solution": "To solve the integral \\( \\int_0^1 (3x^2 + 2x) \\, dx \\), integrate term by term:\n\n\\( \\int 3x^2 \\, dx = x^3 \\),\n\\( \\int 2x \\, dx = x^2 \\).\n\nSo the antiderivative is \\( x^3 + x^2 + C \\).\n\nEvaluating from 0 to 1:\n\\( (1^3 + 1^2) - (0^3 + 0^2) = 1 + 1 = 2 \\).",
            "messages": [
                {
                    "role": "user",
                    "content": "Evaluate the integral \\( \\int_0^1 (3x^2 + 2x) \\, dx \\).",
                },
                {
                    "role": "assistant",
                    "content": "Sure. The answer is \\( \\boxed{2} \\).",
                },
            ],
        }
    ]
    st.session_state.filename = "untitled.jsonl"
    st.session_state.data_loaded = True
    st.sidebar.success("✅ Started new JSONL session with sample entry.")

SERVER_JSONL_DIR = "./results"
server_files = [
    os.path.relpath(os.path.join(root, file), SERVER_JSONL_DIR)
    for root, _, files in os.walk(SERVER_JSONL_DIR)
    for file in files
    if file.endswith(".jsonl")
]

CONFIG_DIR = "./conf/model"
config_files = (
    [f for f in os.listdir(CONFIG_DIR) if f.endswith((".yml", ".yaml"))]
    if os.path.isdir(CONFIG_DIR)
    else []
)

# --- Server‑side JSONL files -------------------------------------------------
selected_server_file = st.sidebar.selectbox(
    "📁 Or select a server file", [""] + server_files
)

# If the user selects a server file that is different from the one currently
# loaded (or nothing is loaded yet), load it immediately – even if a file was
# already loaded before.
if selected_server_file and selected_server_file != st.session_state.get("filename"):
    server_file_path = os.path.join(SERVER_JSONL_DIR, selected_server_file)
    st.success(f"✅ Loaded server file: {selected_server_file}")
    # Clear any previously uploaded file reference so the new server file wins
    uploaded_file = None
    st.session_state.filename = selected_server_file
    with open(server_file_path, "r", encoding="utf-8") as f:
        with jsonlines.Reader(f) as reader:
            st.session_state.data = list(reader)
    st.session_state.data_loaded = True

if uploaded_file is not None:
    st.sidebar.success("✅ File uploaded successfully.")
else:
    st.sidebar.info("📂 Please upload a .jsonl file to begin.")

if uploaded_file is not None and (
    uploaded_file.name != st.session_state.get("filename")
    or not st.session_state.data_loaded
):
    st.session_state.filename = uploaded_file.name
    with jsonlines.Reader(uploaded_file) as reader:
        st.session_state.data = list(reader)
    st.session_state.data_loaded = True

selected_config_file = st.selectbox("🤖 Select LLM model config", [""] + config_files)
if selected_config_file:
    if (
        "apimodel" not in st.session_state
        or st.session_state.get("selected_config_file") != selected_config_file
    ):
        with open(os.path.join(CONFIG_DIR, selected_config_file), "r") as cf:
            config = yaml.safe_load(cf)
        st.session_state.apimodel = APIModel(
            api_key=config["api_key"],
            model=config["model_name_or_path"],
            base_url=config.get("base_url"),
            organization=config.get("organization"),
            project=config.get("project"),
        )
        st.session_state.selected_config_file = selected_config_file

if st.session_state.get("data"):
    selected_index = st.selectbox(
        "🔍 Select Entry Index", list(range(len(st.session_state.data)))
    )
    item = st.session_state.data[selected_index]
    messages = item.setdefault("messages", [])
    st.sidebar.write("🔍 Selected Entry:", selected_index)
    st.sidebar.write("📬 Messages (count):", len(messages))
    with st.expander(f"Entry {selected_index}", expanded=True):
        st.markdown(f"**Index**: `{item.get('index', selected_index)}`")
        st.text_area(
            "Question",
            item.get("question", ""),
            key=f"q_{selected_index}",
            disabled=True,
        )
        st.text_area("Answer", item.get("answer", ""), key=f"a_{selected_index}")
        solution_content = item.get("solution", "")
        sol_lines = solution_content.count("\n") + 1
        sol_height = min(800, max(100, (sol_lines + 2) * 22))
        st.text_area(
            "Solution", solution_content, key=f"s_{selected_index}", height=sol_height
        )

        for m_idx, message in enumerate(messages):
            st.markdown(f"### **Message {m_idx + 1}**")
            role = message.get("role", "unknown")
            role_options = ["system", "user", "assistant"]
            messages[m_idx]["role"] = st.selectbox(
                f"Role {m_idx + 1}",
                role_options,
                index=role_options.index(role) if role in role_options else 0,
                key=f"role_{selected_index}_{m_idx}",
                label_visibility="collapsed",
            )
            if st.button(f"🗑️ Delete", key=f"delete_{selected_index}_{m_idx}"):
                messages.pop(m_idx)
                st.rerun()
            content = message.get("content", "")
            line_count = content.count("\n") + 1
            height = min(800, max(100, (line_count + 2) * 22))
            messages[m_idx]["content"] = st.text_area(
                f"Message {m_idx + 1} content",
                value=content,
                key=f"msg_{selected_index}_{m_idx}",
                height=height,
                label_visibility="collapsed",
            )

        if st.button("➕ Add Message"):
            new_message = {"role": "user", "content": ""}
            messages.append(new_message)
            st.rerun()

        st.markdown("### ⚙️ Generation Settings")
        max_tokens = st.number_input(
            "Max Tokens", min_value=1, max_value=32000, value=4096, step=1
        )
        temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05
        )

        async def async_get_reply():
            if "apimodel" in st.session_state:
                # Use asynchronous generation to avoid blocking UI
                response = await st.session_state.apimodel.agenerate(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    # extra_body={
                    #     "skip_special_tokens": False,
                    #     "continue_final_message": True,
                    #     "add_generation_prompt": False,
                    # },
                )
                # `agenerate` returns a dict; the assistant reply is the last message
                if isinstance(response, dict) and "messages" in response:
                    return response["messages"][-1]["content"]
                # Fallback if unexpected format
                return str(response)
            else:
                st.warning(
                    "⚠️ Please select an LLM model config from the dropdown above."
                )
                return "⚠️ No LLM model selected."

        def handle_get_response():
            st.session_state.getting_response = True

        st.button("🔁 Get Assistant Response", on_click=handle_get_response)

    # Handle LLM assistant response after message-editing block, before expander
    if (
        st.session_state.get("getting_response", False)
        and "response_added" not in st.session_state
    ):
        import nest_asyncio

        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        with st.spinner("⏳ Calling LLM, please wait..."):
            assistant_reply = loop.run_until_complete(async_get_reply())
        # Avoid duplicating: some LLMs modify `messages` in-place
        if not (
            messages
            and messages[-1].get("role") == "assistant"
            and messages[-1].get("content") == assistant_reply
        ):
            messages.append({"role": "assistant", "content": assistant_reply})

        st.session_state.response_added = True
        st.rerun()
    elif st.session_state.get("response_added", False):
        st.session_state.getting_response = False
        del st.session_state["response_added"]

    if st.button("💾 Export edited JSONL"):
        output = StringIO()
        with jsonlines.Writer(output) as writer:
            for idx, item in enumerate(st.session_state.data):
                item["answer"] = st.session_state.get(
                    f"a_{idx}", item.get("answer", "")
                )
                item["solution"] = st.session_state.get(
                    f"s_{idx}", item.get("solution", "")
                )
                writer.write(item)
        st.download_button(
            label="📥 Download JSONL",
            data=output.getvalue(),
            file_name=f"edited_{st.session_state.filename or 'data'}.jsonl",
            mime="application/json",
        )
elif not st.session_state.get("data_loaded", False):
    st.warning("📂 No file uploaded yet. Please select a .jsonl file above.")
