import os

import streamlit as st
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import OpenAI


SYSTEM_INSTRUCTIONS = (
    "You are a helpful AI assistant that answers questions only about computers "
    "and provides concise information. If the user asks about something outside "
    "this domain, politely decline to answer."
)


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
    stream = client.responses.create(
        model=deployment,
        instructions=SYSTEM_INSTRUCTIONS,
        input=prompt,
        previous_response_id=st.session_state.last_response_id,
        stream=True,
    )
    for event in stream:
        if event.type == "response.output_text.delta":
            yield event.delta
        elif event.type == "response.completed":
            st.session_state.last_response_id = event.response.id


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