import streamlit as st
from backend.testVdocument_interaction_endpoint import DOCUMENT_INTERACTION
from langchain_core.messages import HumanMessage, AIMessage
import os
import tempfile
import time
from icecream import ic

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI  # Default LLM (you can replace this with your preferred LLM)
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings  # Default embeddings (replace if needed)


# Constants
LOGO_URL = "KA.jpg"  # Replace with your company logo URL

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_docs" not in st.session_state:
    st.session_state.selected_docs = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "login"

# Initialize backend class
llm_chat_model_name = "gemini-1.5-pro"
llm_embedding_model_name = "text-embedding-004"
doc_interaction = DOCUMENT_INTERACTION(llm_chat_model=llm_chat_model_name, llm_embedding_model=llm_embedding_model_name)

# Login Page
def login_page():
    st.image(LOGO_URL, width=200)
    st.title("Login to Your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "password":  # Replace with actual authentication logic
            st.session_state.logged_in = True
            st.session_state.current_page = "main"
            st.rerun()
        else:
            st.error("Invalid username or password")

# Main Dashboard
def main_dashboard():
    st.title("Welcome to KA-GPT")
    st.write("""
    This app allows you to interact with your documents in a conversational way. 
    You can upload documents, select existing ones, find related documents to your topic, search like a human and chat with them using our advanced AI chatbot.
    """)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Chat with Documents", use_container_width=True):
            st.session_state.current_page = "chat"
            st.rerun()
    with col2:
        if st.button("Find related Documents", use_container_width=True):
            st.session_state.current_page = "find_doc"
            st.rerun()

# Chat with Documents Page
def chat_with_documents():
    st.title("Chat with selected Documents")

    # Left Sidebar for Document Selection
    with st.sidebar:
        st.subheader("Existing Documents")
        existing_docs = doc_interaction.list_exist_vdbs()
        if existing_docs:
            st.write("Select from existing documents or upload new Doc/s to chat with:")
            for doc in existing_docs:
                if st.checkbox(doc):
                    if doc not in st.session_state.selected_docs:
                        st.session_state.selected_docs.append(doc)
                else:
                    if doc in st.session_state.selected_docs:
                        st.session_state.selected_docs.remove(doc)
        else:
            st.write("No documents found.")

        # Upload new documents
        st.subheader("Upload New Documents")
        uploaded_files = st.file_uploader(
            "Upload one or more documents (PDF, DOC, DOCX)",
            accept_multiple_files=True,
            type=["pdf", "doc", "docx"]
        )

        if uploaded_files and st.button("Upload and Process"):
            # Save uploaded files to a temporary directory
            temp_dir = tempfile.mkdtemp()
            saved_file_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_file_paths.append(file_path)

            # Process files with a progress spinner
            with st.spinner("Processing documents..."):
                if doc_interaction.upload_and_save_doc(saved_file_paths):
                    st.success("Documents uploaded and processed successfully!")
                    st.rerun()
                else:
                    st.error("Error processing documents.")

        # Start Chatting Button
        if st.session_state.selected_docs and st.button("Start Chatting", use_container_width=True):
            ic(st.session_state.selected_docs)
            ic(type(st.session_state.selected_docs))
            st.session_state.needed_vectorstore = doc_interaction.retrieve_needed_vdbs(st.session_state.selected_docs)
            ic(st.session_state.needed_vectorstore)
            ic(type(st.session_state.needed_vectorstore))

            st.session_state.chat_history = []
            st.rerun()

    # Chat Interface
    if "needed_vectorstore" in st.session_state:
        st.subheader("Chat Interface")

        # Clear All Conversation Button
        if st.button("Clear All Conversation"):
            st.session_state.chat_history = []  # Reset the entire chat history
            st.rerun()

        # Display chat history
        for message in st.session_state.chat_history:
            if hasattr(message, "content"):
                role = "user" if isinstance(message, HumanMessage) else "assistant"
                with st.chat_message(role):
                    st.write(message.content)

        # Get user input
        user_query = st.chat_input("Ask a question about the selected documents...")
        if user_query:
            with st.chat_message("user"):
                st.write(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))

            # Simulate streaming response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                # Call the backend function to get the response
                response, st.session_state.chat_history = doc_interaction.chat_with_saved_doc(
                    user_query, st.session_state.needed_vectorstore, st.session_state.chat_history
                )
                # Simulate streaming
                for chunk in response["answer"].split():
                    full_response += chunk + " "
                    time.sleep(0.05)  # Simulate delay
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

            st.session_state.chat_history.append(AIMessage(content=response["answer"]))

# Chat Widget
def chat_widget():
    st.title("💬 Chat with Your Documents")
    st.write("Welcome to the chat interface. Start chatting with your selected documents!")

    # Retrieve combined vector store from selected documents
    if "selected_docs" in st.session_state:
        combined_vector_store = doc_interaction.retrieve_needed_vdbs(st.session_state.selected_docs)
    else:
        st.error("No documents selected. Please go back to the search page and select documents.")
        return

    # Initialize conversation memory
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # User input
    user_input = st.text_input("Type your message here and press Enter to chat...")
    if user_input:
        # Generate response using RAG chain
        with st.spinner("Generating response..."):
            response = doc_interaction.chat_with_saved_docV2(user_query=user_input, vector_store=combined_vector_store)

        # Display user input and assistant response
        st.write(f"**You:** {user_input}")
        st.write(f"**Assistant:** {response}")

# Document Search Page
def document_search_page():
    st.title("🔍 Document Search")
    st.write("Enter your query to find related documents.")

    # User input for search query
    user_query = st.text_input("Search for documents:", placeholder="Type your query here...")

    # Perform search when the user submits a query
    if user_query:
        with st.spinner("Searching for related documents..."):
            related_docs = doc_interaction.get_related_documents(user_query)

            if related_docs:
                st.success(f"Found {len(related_docs)} related documents!")
                st.write("Select the documents you want to chat with:")

                # Display selectable document names (using checkboxes)
                selected_docs = []
                for doc in related_docs:
                    if st.checkbox(doc):
                        selected_docs.append(doc)

                # "Start Chat" button
                if selected_docs:
                    if st.button("🚀 Start Chat", help="Click to start chatting with the selected documents"):
                        # Store selected documents in session state for later use
                        st.session_state.selected_docs = selected_docs
                        st.session_state.show_chat = True  # Flag to show the chat widget
                        st.success(f"Selected documents: {', '.join(selected_docs)}")
                else:
                    st.warning("Please select at least one document to start chatting.")
            else:
                st.error("No documents found for your query. Please try again.")
    else:
        st.info("Please enter a search query to begin.")

    # Show the chat widget if the "Start Chat" button has been clicked
    if st.session_state.get("show_chat", False):
        chat_widget()


# Main App Logic
def main():
    if not st.session_state.logged_in:
        login_page()
    else:
        if st.session_state.current_page == "main":
            main_dashboard()
        elif st.session_state.current_page == "chat":
            chat_with_documents()
        elif st.session_state.current_page == "find_doc":
            document_search_page()


if __name__ == "__main__":
    st.set_page_config(page_title="KA-GPT")
    main()
