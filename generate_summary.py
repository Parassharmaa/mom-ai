import sys
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter()

# get transcript file key from args
file_key = sys.argv[1]

# get transcript text
text = open(file_key, "r").read()


llm = OpenAI(temperature=0)

texts = text_splitter.split_text(text)
from langchain.docstore.document import Document

docs = [Document(page_content=t) for t in texts]

chain = load_summarize_chain(llm, chain_type="map_reduce")

output = chain.run(docs)

print(output)



