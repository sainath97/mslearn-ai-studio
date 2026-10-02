import json
import os
import platform
import sys

import psutil
import streamlit as st
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import OpenAI


SYSTEM_INSTRUCTIONS = (
    "You are a helpful AI assistant that answers questions only about computers "
    "and provides concise information. If the user asks about something outside "
    "this domain, politely decline to answer. When asked about this user's system "
    "configuration, use the get_system_info tool and request only the categories "
    "needed to answer. Do not infer system details that the tool did not return."
)

SYSTEM_INFO_CATEGORIES = ["os", "cpu", "memory", "disk", "python"]
SYSTEM_INFO_TOOLS = [
    {
        "type": "function",
        "name": "get_system_info",
        "description": (
            "Read selected, non-secret system configuration details from this "
            "computer. Use only when the user asks about their system."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": SYSTEM_INFO_CATEGORIES,
                    },
                    "minItems": 1,
                }
            },
            "required": ["categories"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


def get_system_info(categories: list[str]) -> dict[str, object]:
    requested = set(categories)
    if not requested or not requested.issubset(SYSTEM_INFO_CATEGORIES):
        raise ValueError("Unsupported system information category.")

    info: dict[str, object] = {}
    if "os" in requested:
        info["os"] = {
            "name": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.machine(),
        }
    if "cpu" in requested:
        info["cpu"] = {
            "model": platform.processor() or "Unknown",
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
        }
    if "memory" in requested:
        memory = psutil.virtual_memory()
        info["memory"] = {
            "total_bytes": memory.total,
            "available_bytes": memory.available,
            "used_percent": memory.percent,
        }
    if "disk" in requested:
        disks = []
        for partition in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(partition.mountpoint)
            except OSError:
                continue
            disks.append(
                {
                    "mountpoint": partition.mountpoint,
                    "total_bytes": usage.total,
                    "free_bytes": usage.free,
                    "used_percent": usage.percent,
                }
            )
        info["disk"] = disks
    if "python" in requested:
        info["python"] = {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "bitness": 64 if sys.maxsize > 2**32 else 32,
        }
    return info


def run_function_tool(name: str, arguments: str) -> str:
    if name != "get_system_info":
        return json.dumps({"error": "Unsupported function."})
    try:
        parsed_arguments = json.loads(arguments)
        categories = parsed_arguments["categories"]
        if not isinstance(categories, list) or not all(
            isinstance(category, str) for category in categories
        ):
            raise ValueError("Categories must be a list of strings.")
        return json.dumps(get_system_info(categories))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return json.dumps({"error": str(error)})


@st.cache_resource
def get_openai_client() -> OpenAI:
    load_dotenv()
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is missing from the .env file.")

    base_url = endpoint
    if not base_url.endswith("/openai/v1"):
        base_url = f"{base_url}/openai/v1"

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://ai.azure.com/.default"
    )
    return OpenAI(base_url=f"{base_url}/", api_key=token_provider)


def stream_answer(client: OpenAI, deployment: str, prompt: str):
    response_stream = client.responses.create(
        model=deployment,
        instructions=SYSTEM_INSTRUCTIONS,
        input=prompt,
        previous_response_id=st.session_state.last_response_id,
        tools=SYSTEM_INFO_TOOLS,
        stream=True,
    )
    while True:
        completed_response = None
        for event in response_stream:
            if event.type == "response.output_text.delta":
                yield event.delta
            elif event.type == "response.completed":
                completed_response = event.response

        if completed_response is None:
            return

        function_calls = [
            item for item in completed_response.output if item.type == "function_call"
        ]
        if not function_calls:
            st.session_state.last_response_id = completed_response.id
            return

        tool_outputs = [
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": run_function_tool(call.name, call.arguments),
            }
            for call in function_calls
        ]
        response_stream = client.responses.create(
            model=deployment,
            instructions=SYSTEM_INSTRUCTIONS,
            previous_response_id=completed_response.id,
            input=tool_outputs,
            tools=SYSTEM_INFO_TOOLS,
            stream=True,
        )


def main() -> None:
    st.set_page_config(page_title="Computer Help Desk")
    load_dotenv()
    deployment = os.getenv("MODEL_DEPLOYMENT", "").strip()

    st.title("Computer Help Desk")
    st.caption("Ask a computer-related question. Responses stream as they arrive.")

    with st.sidebar:
        st.subheader("Session")
        st.caption(f"Model deployment: {deployment or 'Not configured'}")
        if st.button("New chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_response_id = None
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.last_response_id = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about computers...")
    if not prompt:
        return

    if not deployment:
        st.error("MODEL_DEPLOYMENT is missing from the .env file.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(
                stream_answer(get_openai_client(), deployment, prompt)
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
        except Exception as error:
            st.error(f"Unable to get a response: {error}")


if __name__ == "__main__":
    main()