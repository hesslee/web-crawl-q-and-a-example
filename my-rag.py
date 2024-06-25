from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from tqdm import tqdm
#import os
#os.environ['USER_AGENT'] = 'Altibase-RAG'

###### Load ###############################################
def load_directory_md(dir):
    loader = DirectoryLoader(dir, glob="**/*.md", 
                             show_progress=True, 
                             use_multithreading=True,
                             #recursive=False,
                             max_concurrency=4,
                            )
    docs = loader.load()
    return docs

def load_web(url):
    #bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=(url,),
        #bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    return docs

###### Split ###############################################
def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits


def split_into_sublists(original_list, max_elements_per_sublist):
  """
  Splits a list into sublists with a maximum number of elements each.

  Args:
      original_list: The list to split.
      max_elements_per_sublist: The maximum number of elements per sublist.

  Returns:
      A list of sublists, each containing at most max_elements_per_sublist items.
  """

  sublists = []
  current_sublist = []

  for element in original_list:
    # Add the element to the current sublist
    current_sublist.append(element)

    # Check if the sublist has reached the maximum size
    if len(current_sublist) == max_elements_per_sublist:
      # Add the full sublist to the list of sublists
      sublists.append(current_sublist)
      # Start a new sublist
      current_sublist = []

  # Add the remaining elements to the last sublist (if any)
  if current_sublist:
    sublists.append(current_sublist)

  return sublists

###### Embed & Store ###############################################
def embed_store(docs):
    embeddings = OllamaEmbeddings(model="llama3", base_url="http://localhost:8080")

    # See docker command above to launch a postgres instance with pgvector enabled.
    connection = "postgresql+psycopg://postgres:postgres@localhost:5432/opengpts"  # Uses psycopg3!
    collection_name = "altibase_public"

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
        #async_mode=True,
    )

    vectorstore.add_documents(docs,)
    #add_texts() is too slow. 
    #for doc in tqdm(docs, desc="Processing Documents"):
    #    text = doc.page_content
    #    meta = doc.metadata
    #    # Call the add_texts function to process the data
    #    vectorstore.add_texts(text, meta)


docs = load_directory_md("/home/hess/work/alti-documents")
#docs = load_directory_md("/home/hess/work/alti-documents/Technical Documents/kor")
#docs = load_web("https://altibase.com/")
#docs = load_web("https://docs.altibase.com/")
#docs = load_web("http://support.altibase.com/kr/")
#docs = load_web("http://support.altibase.com/en/")

docs = split_docs(docs)
# process 300 docs at a time. if it is too large then PostgreSQL can produce error.
docs = split_into_sublists(docs, 300)
for texts in tqdm(docs, desc="Processing Documents"):
    embed_store(texts)

