from langchain.text_splitter import RecursiveCharacterTextSplitter
from backend.utils.helper import assign_unique_ids


from langchain.text_splitter import MarkdownHeaderTextSplitter

class ContentChunkerClass:
    def __init__(self, text_content:str):
        self._text_content = text_content

    def recursive_chunk(self,
                        chunk_size=3000,
                        chunk_overlap=1000,
                        length_function=len,
                        is_separator_regex=False,
                        separators =["\n\n"],
                        ):
        text_splitter = RecursiveCharacterTextSplitter(
            separators = separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            is_separator_regex=is_separator_regex,
        )
        chunks = text_splitter.split_text(self._text_content)
        ids = assign_unique_ids(chunks)

        return ids, chunks

    def Context_aware_chunk(self):
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        chunks = markdown_splitter.split_text(self._text_content)
        ids = assign_unique_ids(chunks)

        return ids, chunks


# ========================= TESTING ===================
"""
from document_processing.retrieval_pipeline._1content_reading import ContentReaderClass
c = ContentReaderClass(file_paths=["D:/AI Projects/KA-GPT/test_data/SAICO_Network Security Policy_V2.00_POL127.docx"])
nms, cont = c.extract_text_using_docling()

cc = ContentChunkerClass(text_content= cont)
chunks_ids, text_chunks= cc.recursive_chunk()

# print(f"||||||||||||||||||{chunks_ids}|||||||||||||||||\n\n")

# print(f"||||||||||||||||||{text_chunks}|||||||||||||||||\n\n")

def print_id_content_pairs(_ids:list, _contents:list):
    if len(_ids) != len(_contents):
        raise ValueError("Number of IDs and contents must be the same.")

    for i in range(len(_ids)):
        print("\n******** NEW CHUNK *********\n")
        print(f"ID: {_ids[i]} \n | Content:\n {_contents[i]}")


print_id_content_pairs(chunks_ids, text_chunks)
"""


