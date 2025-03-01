from langchain_community.document_loaders import DirectoryLoader

REBUILD = True

if REBUILD:
    loader = DirectoryLoader("../../benchmark/ott", glob="**/*.pdf")
    books = loader.load()
    print(len(books))

from langchain_text_splitters import RecursiveCharacterTextSplitter

if REBUILD:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(books)
    all_splits = all_splits[19:23]
    #for chunk in all_splits:
    #    print(chunk)

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

from langchain import hub
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = OllamaLLM(model="llama3")

retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = hub.pull("rlm/rag-prompt")
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

question = "What is the royalty income in FY2023"
#question = "What is the office of Technology transfer"
#question = "In FY2023, how many licensed products provided royalty income back to the NIH"
#question = "Where was royalty income mentioned in the document"
#question = "Where is OTT mentioned in the document?"
#question = "What did NCATS do? Show me the numbers of inventions and patents"
#question = "In the first paragraph, the document mentioned about royal income. How much was the dollar amount of the royal income?"

#question = "how many licensed products developed from NIH, what are their names"
print(question)
answers = qa_chain.invoke(question)


print(answers)
