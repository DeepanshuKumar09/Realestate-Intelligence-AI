
import json
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(
    page_title="Real Estate Intelligence",
    page_icon="🏢",
    layout="wide"
)

# --------------------------------------------------
# PROFESSIONAL STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #8b949e;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    [data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    .source-heading {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION MEMORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🏢 Real Estate AI")

    st.caption("Intelligent property and market research")

    st.divider()

    st.subheader("Knowledge Base")

    st.markdown("""
    **DarGlobal**

    Company information, developments,
    projects and business updates.

    **Wasalt**

    Property listings, locations
    and real-estate information.
    """)

    st.divider()

    st.subheader("Capabilities")

    st.markdown("""
    - Conversational AI
    - Retrieval-Augmented Generation
    - Semantic document search
    - Streaming responses
    - Source-backed answers
    """)

    st.divider()

    if st.button(
        "Clear Conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    st.caption("Powered by LangChain · ChromaDB · OpenRouter")


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">Real Estate Intelligence Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about DarGlobal developments, '
    'company information and Wasalt property listings.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# DISPLAY SOURCES
# --------------------------------------------------

def display_sources(sources):

    if not sources:
        return

    with st.expander("References & Sources"):

        seen_urls = set()

        for source in sources:

            url = source.get("url")
            title = source.get("title", "View reference")
            website = source.get("source", "")

            if url and url not in seen_urls:

                seen_urls.add(url)

                st.markdown(f"**{title}**")

                if website:
                    st.caption(website)

                st.markdown(f"[Open reference]({url})")

                st.divider()


# --------------------------------------------------
# DISPLAY CONVERSATION HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_sources(message.get("sources", []))


# --------------------------------------------------
# STREAMING RESPONSE
# --------------------------------------------------

def stream_response(question, history):

    collected_sources = []

    with requests.post(
        API_URL,
        json={
            "question": question,
            "history": history
        },
        stream=True,
        timeout=(10, 180)
    ) as response:

        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):

            if not line:
                continue

            event = json.loads(line)

            if event["type"] == "sources":

                collected_sources = event.get("data", [])

            elif event["type"] == "token":

                token = event.get("data", "")

                if token:
                    yield token

            elif event["type"] == "done":
                break

    st.session_state.current_sources = collected_sources


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask about DarGlobal or Wasalt..."
)


if question:

    # Show user message

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Previous conversation for memory

    history = st.session_state.messages[:-1]

    # Generate answer

    with st.chat_message("assistant"):

        st.session_state.current_sources = []

        try:

            with st.spinner("Analyzing information..."):

                answer = st.write_stream(
                    stream_response(
                        question,
                        history[-10:]
                    )
                )

            sources = st.session_state.current_sources

            display_sources(sources)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except requests.RequestException as error:

            st.error(
                f"Backend connection failed: {error}"
            )

        except Exception as error:

            st.error(
                f"An unexpected error occurred: {error}"
            )
