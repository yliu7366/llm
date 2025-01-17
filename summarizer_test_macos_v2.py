import os
import shutil

from langchain_community.document_loaders import DirectoryLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_ollama.llms import OllamaLLM

# macOS specific
os.environ['TESSDATA_PREFIX'] = '/opt/miniconda3/envs/llm-conda/share/tessdata'

MODEL = 'llama3-chatqa'

def loadDocuments():
  loader = DirectoryLoader("../../benchmark/ott", glob="**/*.pdf", use_multithreading=True)
  books = loader.load()
  print('Number of documents:', len(books))
  return books

def textSplitter(books):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  all_splits = text_splitter.split_documents(books)
  print('Number of chunks:', len(all_splits))
  #all_splits = all_splits[:2233]
  #for chunk in all_splits:
  #  print(chunk.page_content)
  return all_splits

def semanticSplitter(books):
  #text = []
  #for book in books:
  #  text.append(book.page_content)

  text = [books[0].page_content]
  text_splitter = SemanticChunker(OllamaEmbeddings(model=MODEL))
  docs = text_splitter.create_documents(text)
  print('Number of chunks:', len(docs))
  return docs

documents = loadDocuments()
chunks = textSplitter(documents)
#chunks = semanticSplitter(documents)

vectorstore = Chroma.from_documents(
  documents=chunks,
  embedding=OllamaEmbeddings(model=MODEL),
)

# prepare retrievers
vectorstore_retreiver = vectorstore.as_retriever()
bm25_retriever = BM25Retriever.from_documents(chunks)
ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver,
                                                   bm25_retriever],
                                       weights=[0.3, 0.7])

llm = OllamaLLM(model=MODEL)

def generate_response(query, retrieved_docs):
  context = " ".join(doc.page_content for doc in retrieved_docs)
  prompt = f"Based on the following information: {context}\n\nAnswer the query: {query}"
  response = llm.invoke(prompt)
  return response

def rag_system(query):
  #retrieved_docs = vectorstore.similarity_search(query, k=10) # Retrieve documents based on the query
  retrieved_docs = ensemble_retriever.invoke(query)
  print('\nQuery results:')
  for doc in retrieved_docs:
    print("***", doc.page_content)
  response = generate_response(query, retrieved_docs) # Generate an answer using the language model
  return response

question = "How much was the royalty income"
#question = "How many licensed products mentioned in the document"
#question = "What is the office of Technology transfer"
#question = "In FY2023, how many licensed products provided royalty income back to the NIH"
#question = "Where was royalty income mentioned in the document"
#question = "Where is OTT mentioned in the document?"
#question = "What did NCATS do? Show me the numbers of inventions and patents"
#question = "In the first paragraph, the document mentioned about royal income. How much was the dollar amount of the royal income?"

#question = "Find each year's royalty income and list them by year"
print("Question:", question)
answers = rag_system(question)
print("\nAnswer:", answers)
