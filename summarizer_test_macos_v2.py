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
  prompt = f"You are an assistant for question-answering tasks. Based on the following pieces of retrieved context to answer the question.\nContext: {context}\nQuestion: {query}\nAnswer:"
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

#question = "List each year's royalty income. If one year doesn't have royalty income, say no royalty income for that year"
print("Question:", question)
answers = rag_system(question)
print(MODEL+":\nAnswer:", answers)

"""
llama3.3:70b            a6eb4748fd29    42 GB
llama3-chatqa:latest    b37a98d204b2    4.7 GB
llama3:latest           365c0bd3c000    4.7 GB
llama3.2:latest         a80c4f17acd5    2.0 GB

Question = "List each year's royalty income. If one year doesn't have royalty income, say no royalty income for that year"

NotebookLM
Here is a list of the royalty income for each fiscal year mentioned in the sources:
FY2021: $127.6 million
FY2022: $704 million
FY2023: $639 million

llama3.3:70b: 
Here are the royalty incomes listed for each year:

* FY2022: $704 million in royalty income
* FY2023: $639 million in royalty income 

Note: There is no mention of royalty income for other years (e.g. FY2021), so I won't include those years in the list since there's "no royalty income" mentioned for them.

llama3-chatqa:
Answer:  The NIH received $704 million in royalty income in FY2022. There was no royalty income reported for FY2021 or prior years. In FY2023, the NIH received $639 million in royalty income

llama3:
Answer: Based on the provided context, here are the answers to your question:

* FY2022: $704 million
* FY2023: $639 million

llama3.2:
Answer: Here are the years' royalty incomes listed:

* FY2022: $704 million
* FY2023: $639 million

"""
