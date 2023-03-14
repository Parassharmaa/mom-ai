import sys
from langchain import OpenAI, PromptTemplate, LLMChain
import multiprocessing
from langchain.text_splitter import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter()

# get transcript file key from args
file_key = sys.argv[1]

# get transcript text
text = open(file_key, "r").read()


llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")

texts = text_splitter.split_text(text)
from langchain.docstore.document import Document

prompt = PromptTemplate(
    input_variables=['context'],
   template="Identify the keypoints for meeting minutes in the following: {context} \n\n Key points:\n-",
)

docs = [Document(page_content=t) for t in texts]


key_points = []

def get_key_points(doc):
    chain = LLMChain(llm=llm, prompt=prompt, verbose=1)
    res = chain.run(doc.page_content)
    return res

if __name__ == "__main__":
    with multiprocessing.Pool(processes=8) as pool:
        key_points = pool.starmap(
            get_key_points, zip(docs)
        )
        pool.close()
        pool.join()

    print(key_points)

    prompt_mom = PromptTemplate(
        input_variables=['key_points'],
        template="Below are the pointers for a meeting. Generate a meeting minutes include section for key things discussed and action items accordingly.\n{key_points}.",
    )

    chain = LLMChain(llm=llm, prompt=prompt_mom, verbose=1)
    mom = chain.run(''.join(key_points))

    print(mom)


    # save mom to file
    file_name = file_key.split("/")[-1].split(".")[0]

    with open(f"{file_name}_mom.txt", "w") as f:
        f.write(mom)
        f.close()


