import PyPDF2
import docx
import os
# ------------------------------------------------------
from docling.document_converter import DocumentConverter
converter = DocumentConverter()
# ------------------------------------------------------

class ContentReaderClass:
    def __init__(self, file_paths:list, context_threshold_ratio:float = 0.60):
        self.threshold_ratio = context_threshold_ratio
        self.file_paths = file_paths

    def extract_text_normal(self):
        """Extracts text from PDF and Word documents.
        Args:
            file_paths: A list of file paths to process
        Returns:
            A list of extracted text strings, one for each file.
        """
        text_list = []
        for file_path in self.file_paths:
            try:
                if file_path.endswith(".pdf"):
                    with open(file_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page_num in range(len(reader.pages)):
                            page = reader.pages[page_num]
                            text += page.extract_text()
                        text_list.append(text)

                elif file_path.endswith(".docx"):
                    doc = docx.Document(file_path)
                    fullText = []
                    for para in doc.paragraphs:
                        fullText.append(para.text)
                    text_list.append('\n'.join(fullText))

                else:
                    raise ValueError("Unsupported file format. Please upload PDF or Word documents.")
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                text_list.append(f"Error processing file {file_path}")

            document_content = "\n\n".join(text_list)

            # ic(type(text_list))
            # ic(document_content)

            return len(text.split()), document_content

    def extract_text_using_docling(self, return_document_as="merged_content"):
        document_names = []
        counter = 1
        if return_document_as == "merged_content":
            documents_content_merged = ""
            for file_path in self.file_paths:
                # getting document names to be used as index in VDB
                filename = os.path.splitext(os.path.basename(file_path))[0]
                document_names.append(filename)
                # reading each document content
                result = converter.convert(file_path)
                doc_content = f"\n\n_________________________________________________ Document Number: {str(counter)} _________________________________________________\n" + result.document.export_to_markdown()

                # print(doc_content)
                # print("\n ============================ \n")
                documents_content_merged += doc_content + f"""
                   \n\n
                   \n␃: End of Document {str(counter)} Content.......................................
                   \n______________________________________________________________________________________________________
                   """
                counter += 1
            return document_names, documents_content_merged

        elif return_document_as == "isolated_content":
            documents_content_isolated = []
            for file_path in self.file_paths:
                # getting document names to be used as index in VDB
                filename = os.path.splitext(os.path.basename(file_path))[0]
                document_names.append(filename)
                # reading each document content
                result = converter.convert(file_path)
                doc_content = f"\n\n_________________________________________________ Document Number: {str(counter)} _________________________________________________\n" + result.document.export_to_markdown()

                # print(doc_content)
                # print("\n ============================ \n")
                documents_content_isolated.append(doc_content + f"""
                   \n\n
                   \n␃: End of Document {str(counter)} Content.......................................
                   \n______________________________________________________________________________________________________
                   """)
                counter += 1
            return document_names, documents_content_isolated

    def tokens_condition(self, text_length, model_name):
        """
        this function check if the input document text content will be converted to vectorDB or not to be retrieved.
        Args:
        text_length (int): The length of the text in tokens.
        model_name (str): The name of the language model.

        :return:
        - bool flag of true: if the text will be passed as it is
        - bool flag of false: if the text will be passed to the vector DB pipeline
        """
        model_context_windows = {
            "gemini-1.5-pro": 2000000,
            "gemini-1.5-flash": 1000000,
            "gemini-1.5-flash-8b": 1000000,
            "gpt-3.5-turbo": 16000,
            "gpt-3.5-turbo-0125": 16000,
            "gpt-4": 8000,
            "gpt-4-turbo":128000,
            "gpt-4o":128000
        }
        if model_name not in model_context_windows:
            raise ValueError(f"Model '{model_name}' not found in the lookup dictionary.")

        context_window = model_context_windows[model_name]
        # print(type(threshold_ratio))
        # print(type(self.threshold_ratio))

        threshold = self.threshold_ratio * context_window

        return text_length <= threshold

# =================================================================================================================
"""

ins = ContentReaderClass([r"D:\AI Projects\KA-GPT\test_data\4. End to End Project.pdf", r"D:/AI Projects/KA-GPT/test_data/sample3.docx"])
names, cont = ins.extract_text_using_docling(return_document_as="merged_content")
# from icecream import ic

print(names)
print(cont)
"""