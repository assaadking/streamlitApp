# importing related modules...
from sqlalchemy.testing.suite.test_reflection import metadata

from backend.document_processing.retrieval_pipeline._1content_reading import ContentReaderClass
from backend.document_processing.retrieval_pipeline._2content_chunking import ContentChunkerClass
from backend.utils.helper import clean_for_database_filename, find_faiss_files
from dotenv import load_dotenv
import os
from icecream import ic

# importing required endpoints
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv('D:/PyCharmProjects/documentChat/backend/utils/.env_var')

vector_db_path = os.getenv("vector_db_path")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
contextualize_q_system_prompt = os.getenv("contextualize_q_system_prompt")

import os

# Suppress logging warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"


class DOCUMENT_INTERACTION:
    def __init__(self, llm_chat_model: str, llm_embedding_model: str):
        # ------------------------------------------------------------
        # Chat Models
        if llm_chat_model == "gemini-2.0-flash-exp" or llm_chat_model == "gemini-1.5-flash" or llm_chat_model == "gemini-1.5-pro":
            self._llm_chat_model = ChatGoogleGenerativeAI(model=llm_chat_model,
                                                          temperature=0.2,
                                                          max_tokens=None,
                                                          timeout=None,
                                                          max_retries=3,
                                                          api_key=GOOGLE_API_KEY
                                                          )
        elif llm_chat_model == "gpt-4-turbo" or llm_chat_model == "gpt-4" or llm_chat_model == "gpt-3.5-turbo-0125":
            self._llm_chat_model = ChatOpenAI(model=llm_chat_model,
                                              temperature=0,
                                              max_tokens=None,
                                              timeout=None,
                                              max_retries=2,
                                              api_key=OPENAI_API_KEY,
                                              )
        # Embedding Models
        if llm_embedding_model == "text-embedding-ada-002" or llm_embedding_model == "text-embedding-3-small" or llm_embedding_model == "text-embedding-3-large":
            self._embedding_model = OpenAIEmbeddings(model=llm_embedding_model, api_key=OPENAI_API_KEY)
        elif llm_embedding_model == "text-embedding-004":
            self._embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",
                                                                 api_key=GOOGLE_API_KEY)
        # ------------------------------------------------------------
        self._vector_db_path = vector_db_path
        self._contextualize_q_system_prompt = contextualize_q_system_prompt
        self._system_template = """
                    Your Name is: KA-GPT, built by AI Engineering team in KHATIB & ALAMI COMPANY, You are an assistant for question-answering tasks over a provided documents. 
                    Use the following pieces of retrieved context to answer the question. 
                    If you don't know the answer, just say that you don't know. 
                    
                    When responding to a user question, consider the following:
                        1.Relevance: Directly address the core question or request.
                        2.Clarity: Use simple language and avoid technical jargon.
                        3.Conciseness: Provide a focused answer without unnecessary details.
                        4.Comprehensiveness: Ensure the response is complete and informative.
                        5.Contextual Understanding: Reference specific information from the document to support your answer.
                        6.User-Friendliness: Tailor your response to the user's level of understanding.
                        7.Politeness: Maintain a respectful and helpful tone.
                        
                        And Avoid the following:  
                            1.Providing overly long or complex answers.
                            2.Making assumptions about the user's knowledge.
                            3.Giving irrelevant or misleading information.
                            
                    
                    Context:
                    {context}
                    
                    Answer:
                    """
        self._llm_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_template),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    # ---------------------------------------------------------------------------------
    def upload_and_save_doc(self, uploaded_documents):
        """
          Uploads and saves user documents to a local vector database.

          This method processes user-uploaded documents by:

          1. **Reading Document Content:** Extracts text content from the uploaded documents using the `ContentReaderClass`.
          2. **Chunking Documents:** Splits the extracted text into smaller, manageable chunks using the `ContentChunkerClass`.
          3. **Generating Embeddings:** Creates embeddings for each document chunk using the specified embedding model (`self._embedding_model`).
          4. **Building Vector Store:** Constructs a local FAISS vector store to efficiently store and search the document embeddings.
          5. **Saving Vector Store:** Saves the generated vector store to the specified local path (`self._vector_db_path`) under the user's name (`self._user_name`).

          Returns:
            True if the vector store is successfully saved, False otherwise.
          """
        ic("uploading documents....")
        # read documents content
        ic("Reading uploaded documents......")
        up_document_names, up_document_content = ContentReaderClass(
            file_paths=uploaded_documents).extract_text_using_docling(return_document_as="isolated_content")

        if len(up_document_names) != len(up_document_content):
            raise ValueError("Document names and contents lists must have the same length.")

        ic("Chunking uploaded documents...")
        for name, content in zip(up_document_names, up_document_content):
            _cleaned_doc_name = clean_for_database_filename(name)

            print(f"Document Name: {clean_for_database_filename(name)}")
            print(f"Content:\n{content}\n")
            # chunking uploaded document one by one
            chunks_ids, up_document_chunks = ContentChunkerClass(text_content=content).recursive_chunk()
            # build a new temp vector store
            # inject metadata
            metadata_ = [{"document_name": name, "vector_store_name": _cleaned_doc_name} for _ in
                         range(len(up_document_chunks))]
            # ***********************************************************************
            ic("Creating Vector Store....")
            vector_store = FAISS.from_texts(texts=up_document_chunks,
                                            embedding=self._embedding_model,
                                            ids=chunks_ids,
                                            metadatas=metadata_)
            # save vector store locally
            if vector_store:
                ic("Saving Vector SB locally...... :) ")
                vector_store.save_local(folder_path=self._vector_db_path,
                                        index_name=_cleaned_doc_name)
                print(f">>\nVector Database is successfully Done for: {_cleaned_doc_name}\n")

        return True

    # ---------------------------------------------------------------------------------
    def list_exist_vdbs(self):
        """
        Lists existing vector databases stored as .faiss files.

        This method searches for .faiss files within the specified directory
        and returns the names of the corresponding vector databases without the .faiss extension.
        """
        list_of_current_VDBs_paths = find_faiss_files(self._vector_db_path)
        if list_of_current_VDBs_paths:
            # print("Found the following stored documents:\n")
            document_names = []
            for file_path in list_of_current_VDBs_paths:
                # Extract the filename without the .faiss extension
                filename = os.path.splitext(os.path.basename(file_path))[0]
                # print(filename)
                document_names.append(filename)
            ic(document_names)
            return document_names
        else:
            print("No .faiss files found in the specified directory.")
            return False

    # ---------------------------------------------------------------------------------
    def retrieve_needed_vdbs(self, needed_docs: list):
        """Retrieves and merges vector databases (VDBs) for the specified documents.

        This method loads one or more FAISS vector databases from local storage,
        merging them into a single `FAISS` object.  It efficiently handles the case
        where only one document's VDB is needed, avoiding unnecessary merging.

        Args:
            needed_docs: A list of document names (strings) for which to retrieve
                         the corresponding vector databases.  Each name is expected
                         to correspond to a separate FAISS index file stored locally.

        Returns:
            A `FAISS` object containing the merged vector database.  If only one
            document name is provided, the function returns the corresponding VDB
            directly without merging.

        Raises:
            FileNotFoundError: If any of the specified document names do not correspond
                               to a locally stored FAISS index.  (This is not
                               explicitly handled in the code but is a potential
                               runtime error condition).
            Exception: If there is an issue loading or merging the FAISS indices,
                       such as incompatibility between indices or corruption of the
                       stored data. (This is also not explicitly handled but a
                       potential runtime error.)

        """
        if len(needed_docs) == 1:
            # load vector store
            _loaded_vector_store = FAISS.load_local(folder_path=self._vector_db_path,
                                                    index_name=str(needed_docs[0]),
                                                    embeddings=self._embedding_model,
                                                    allow_dangerous_deserialization=True)
            # ic(needed_docs[0])

        else:
            # ic(needed_docs[0])
            _loaded_vector_store = FAISS.load_local(folder_path=self._vector_db_path,
                                                    index_name=str(needed_docs[0]),
                                                    embeddings=self._embedding_model,
                                                    allow_dangerous_deserialization=True)
            for _doc_name in needed_docs[1:]:
                # ic(_doc_name)
                _vdb = FAISS.load_local(folder_path=self._vector_db_path,
                                        index_name=_doc_name,
                                        embeddings=self._embedding_model,
                                        allow_dangerous_deserialization=True)
                _loaded_vector_store.merge_from(_vdb)

        return _loaded_vector_store

    # ---------------------------------------------------------------------------------
    def chat_with_saved_doc(self, user_query: str, needed_vectorstore, chat_history=[]):
        """
        Engages in a conversational exchange with the user, leveraging a Retrieval Augmented Generation (RAG) pipeline
        to answer questions based on a selection of pre-indexed documents.  Maintains and updates chat history for context.

        This function orchestrates the RAG process, retrieving relevant documents, constructing a question-answering chain,
        and incorporating chat history for contextual awareness.

        Args:
            user_query: The user's question or prompt as a string.
            chat_history: A list of previous messages in the conversation, represented as `HumanMessage` and `AIMessage`
                         objects. Defaults to an empty list, initiating a new conversation.

        Returns:
            A tuple containing:
                - A dictionary representing the response, typically including the "answer" key with the model's generated response.
                - The updated chat history list, including the current user query and the AI's response.

        Raises:
            ValueError: If `needed_docs` is empty or contains invalid document identifiers.  (Consider adding specific exception types)
            Exception: Any exceptions raised during vectorstore retrieval, chain execution, or LLM interaction. (Consider adding specific exception types)

        Details:
            1. **Document Retrieval:** Loads a vector database based on the `needed_docs` and uses it to retrieve the most relevant documents
               for the given `user_query` using similarity search.  Retrieves the top 4 relevant documents (configurable via `search_kwargs={"k": 4}`).

            2. **QA Chain Construction:** Creates a question-answering chain using the specified large language model (`self._llm_chat_model`) and prompt (`self._llm_prompt`).
               This chain is responsible for generating answers based on the retrieved documents.

            3. **History-Aware Retrieval:** Implements a history-aware retriever to incorporate the conversation's context into the document retrieval process.
               This ensures that subsequent queries can reference previous exchanges.

            4. **RAG Pipeline Execution:** Combines the history-aware retriever and the question-answering chain into a complete RAG pipeline.

            5. **Chat History Management:** Appends the user's query and the AI's response to the `chat_history` list, preserving the conversation's context for future interactions.  Uses `HumanMessage` and `AIMessage` objects for structured history.
        """
        # load vector store
        _loaded_vectors = needed_vectorstore

        # ===== QA Chain ======
        _retriever = _loaded_vectors.as_retriever(search_type="similarity",
                                                  search_kwargs={"k": 4})
        # Create document chain
        question_answer_chain = create_stuff_documents_chain(self._llm_chat_model, self._llm_prompt)
        # Create retrieval chain by combining retriever and document chain
        # ------
        history_aware_retriever = create_history_aware_retriever(
            self._llm_chat_model, _retriever, self.contextualize_q_prompt
        )

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        # Handling chat history
        # chat_history = []
        response = rag_chain.invoke({"input": user_query, "chat_history": chat_history})

        chat_history.extend(
            [
                HumanMessage(content=user_query),
                AIMessage(content=response["answer"]),
            ]
        )

        return response, chat_history

    # ---------------------------------------------------------------------------------
    def chat_with_saved_docV2(self, user_query: str, vector_store):
        """
        Engages in a conversational exchange with the user, leveraging a Retrieval Augmented Generation (RAG) pipeline
        to answer questions based on a selection of pre-indexed documents.  Maintains and updates chat history for context.

        This function orchestrates the RAG process, retrieving relevant documents, constructing a question-answering chain,
        and incorporating chat history for contextual awareness.

        Args:
            user_query: The user's question or prompt as a string.
            needed_docs: A list of document identifiers specifying which documents to consider for retrieval.
            chat_history: A list of previous messages in the conversation, represented as `HumanMessage` and `AIMessage`
                         objects. Defaults to an empty list, initiating a new conversation.

        Returns:
            A tuple containing:
                - A dictionary representing the response, typically including the "answer" key with the model's generated response.
                - The updated chat history list, including the current user query and the AI's response.

        Raises:
            ValueError: If `needed_docs` is empty or contains invalid document identifiers.  (Consider adding specific exception types)
            Exception: Any exceptions raised during vectorstore retrieval, chain execution, or LLM interaction. (Consider adding specific exception types)

        Details:
            1. **Document Retrieval:** Loads a vector database based on the `needed_docs` and uses it to retrieve the most relevant documents
               for the given `user_query` using similarity search.  Retrieves the top 4 relevant documents (configurable via `search_kwargs={"k": 4}`).

            2. **QA Chain Construction:** Creates a question-answering chain using the specified large language model (`self._llm_chat_model`) and prompt (`self._llm_prompt`).
               This chain is responsible for generating answers based on the retrieved documents.

            3. **History-Aware Retrieval:** Implements a history-aware retriever to incorporate the conversation's context into the document retrieval process.
               This ensures that subsequent queries can reference previous exchanges.

            4. **RAG Pipeline Execution:** Combines the history-aware retriever and the question-answering chain into a complete RAG pipeline.

            5. **Chat History Management:** Appends the user's query and the AI's response to the `chat_history` list, preserving the conversation's context for future interactions.  Uses `HumanMessage` and `AIMessage` objects for structured history.
        """
        # load vector store
        _loaded_vectors = vector_store

        # ===== QA Chain ======
        _retriever = _loaded_vectors.as_retriever(search_type="similarity",
                                                  search_kwargs={"k": 4})

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_template),
                ("human", "{input}"),
            ]
        )
        llm = self._llm_chat_model
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(_retriever, question_answer_chain)
        response = rag_chain.invoke({"input": user_query})

        return response["answer"]

    # ---------------------------------------------------------------------------------
    def get_related_documents(self, user_query: str):
        """Retrieves a list of related document names based on a user query.

        This function searches for relevant documents within specified vector databases using two distinct retrieval methods:
        Maximal Marginal Relevance (MMR) and similarity score thresholding.  It combines the results from both methods to
        maximize recall while maintaining reasonable precision.

        Args:
            user_query: The user's search query as a string.
            needed_docs: A list of document names corresponding to the vector databases to be searched.  This allows for
                         targeted searches across specific subsets of documents.

        Returns:
            A list of unique document names that are considered related to the user query. Returns `False` if no
            related documents are found.

        Details:
            1. **Vector Database Loading:**  Loads the required vector databases specified in `needed_docs` using the
               `retrieve_needed_vdbs` method.  This assumes that `self.retrieve_needed_vdbs` handles the loading and
               caching of vector databases for efficiency.

            2. **MMR Retrieval:**  Employs Maximal Marginal Relevance (MMR) to retrieve the top 8 most relevant documents,
               balancing relevance and diversity.  The `lambda_mult` parameter controls the trade-off between these two
               factors (higher values emphasize diversity).

            3. **Similarity Score Thresholding Retrieval:**  Uses cosine similarity to retrieve the top 4 documents with
               the highest similarity scores to the user query. This focuses on documents with strong semantic overlap.

            4. **Result Combination and Deduplication:** Combines the results from both MMR and similarity search and removes
               duplicate document names, providing a consolidated list of related documents.

            5. **Return Value:** Returns the list of unique related document names. If no related documents are found by either
               method, the function returns `False`.

        """
        # load vector store
        all_existing_docs = self.list_exist_vdbs()
        compact_vector_store = self.retrieve_needed_vdbs(needed_docs=all_existing_docs)

        _loaded_vectors = compact_vector_store
        # ===== search methods to get related chunks ======
        # 1 - mmr
        _retriever_mmr = _loaded_vectors.as_retriever(search_type="mmr",
                                                      search_kwargs={'k': 8, 'lambda_mult': 9}
                                                      )

        _related_docs_mmr = _retriever_mmr.invoke(user_query)
        # ic(_related_docs_mmr)
        mmr_related_docs_names = list(set([doc.metadata['vector_store_name'] for doc in _related_docs_mmr]))
        # ic(mmr_related_docs_names)

        # 2 - similarity_score_threshold
        _retriever_sct = _loaded_vectors.as_retriever(search_type="similarity",
                                                      search_kwargs={"k": 4}
                                                      )

        _related_docs_sct = _retriever_sct.invoke(user_query)
        # ic(_related_docs_sct)
        sct_related_docs_names = list(set([doc.metadata['vector_store_name'] for doc in
                                           _related_docs_sct]))  # metadata are [ document_name, vector_store_name ]

        _final_related_docs_names = list(set(mmr_related_docs_names + sct_related_docs_names))

        if len(_final_related_docs_names) > 0:
            return _final_related_docs_names
        else:
            return False


# ------------------------------------------------ TEST ----------------------------------------------------------------
test_ins = DOCUMENT_INTERACTION(llm_chat_model="gemini-1.5-pro",
                                llm_embedding_model="text-embedding-004")

# print(test_ins.upload_and_save_doc(uploaded_documents=[r"D:\AI Projects\KA-GPT\test_data\Palestine_IsraelDoc.pdf",
#                                                        r"D:\AI Projects\KA-GPT\test_data\4. End to End Project.pdf",
#                                                        r"D:\AI Projects\KA-GPT\test_data\SAMPLE HR MANUAL.pdf",
#                                                        r"D:\AI Projects\KA-GPT\test_data\Ahmed_Hefnawy_.pdf"]))


# ic(test_ins.list_exist_vdbs())


# print("\n\n __________ TESTING_____________\n\n")
# response , chat_history = test_ins.chat_with_saved_doc(user_query="what do you know about Nakba?",
#                                                        needed_docs=["4.EndtoEndProject",
#                                                                     "Palestine_IsraelDoc",
#                                                                     "Ahmed_Hefnawy_",
#                                                                     "SAMPLEHRMANUAL"])
#
#
# for k, v in response.items():
#     print(f"{k} :: {v} \n\n")
#
# print(response)
# print("\n \n------------ final response ------------\n")
# print(response['answer'])
# print("**********History************\n")
# print(chat_history)

