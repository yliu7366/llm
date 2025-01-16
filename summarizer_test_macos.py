import os

# macOS specific
os.environ['TESSDATA_PREFIX'] = '/opt/miniconda3/envs/llm-conda/share/tessdata'
#os.environ['LANGSMITH_TRACING'] = "false"
#os.environ['LANGSMITH_TRACING_V2'] = "false"

from langchain_community.document_loaders import DirectoryLoader

REBUILD = True

if REBUILD:
    loader = DirectoryLoader("../../benchmark/ott", glob="**/*.pdf")
    books = loader.load()
    print('Number of books:', len(books))

from langchain_text_splitters import RecursiveCharacterTextSplitter

if REBUILD:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    all_splits = text_splitter.split_documents(books)
    all_splits = all_splits[9:13]
    for chunk in all_splits:
        print(chunk.page_content)

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

if REBUILD:
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=OllamaEmbeddings(model="llama3"),
        persist_directory="./chroma_db",
    )
else:
    vectorstore = Chroma(persist_directory='./chroma_db',
                  embedding_function=OllamaEmbeddings(model="llama3"))

from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model="llama3")

def generate_response(query, retrieved_docs):
    context = " ".join(doc.page_content for doc in retrieved_docs)
    prompt = f"Based on the following information: {context}\n\nAnswer the query: {query}"
    response = llm.invoke(prompt)
    return response

def rag_system(query):
  retrieved_docs = vectorstore.similarity_search(query, k=3) # Retrieve documents based on the query
  #print(retrieved_docs)
  response = generate_response(query, retrieved_docs) # Generate an answer using the language model
  return response

question = "What is the royalty income in FY2023"
#question = "What is the office of Technology transfer"
#question = "In FY2023, how many licensed products provided royalty income back to the NIH"
#question = "Where was royalty income mentioned in the document"
#question = "Where is OTT mentioned in the document?"
#question = "What did NCATS do? Show me the numbers of inventions and patents"
#question = "In the first paragraph, the document mentioned about royal income. How much was the dollar amount of the royal income?"

#question = "how many licensed products developed from NIH, what are their names"
print(question)
answers = rag_system(question)


print(answers)
