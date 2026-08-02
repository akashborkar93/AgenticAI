from pprint import pp
from langchain_community.document_loaders import TextLoader
from pathlib import Path

file_path = Path(__file__).resolve().parent / "sample_rag_knowledge_base.txt"

print(file_path)
print(file_path.exists())

loader = TextLoader(str(file_path))

documents = loader.load()

# print(documents[0].metadata)

pp(documents[0].page_content)



