import numpy as np
import openai
import pandas as pd
import pickle
import tiktoken
import time
import ast
import PyPDF2
import tiktoken
openai.api_key = "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"
COMPLETIONS_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]
def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    total={}
    for idx,r in df.iterrows():
        value=get_embedding(r.content)
        time.sleep(1)
        total[idx]=value
    return total
    #total=
    #return {
        #idx: get_embedding(r.title) for idx, r in df.iterrows()
    #}
def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.
    
    fname is the path to a CSV with exactly these named columns: 
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """
    
    df = pd.read_csv(fname, header=0)
    max_dim = max([int(float(c)) for c in df.columns if c != "title" and c != "heading"])
    return {
           (r.title): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }
def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities
def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame , cur:str) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
    count=0
    num=0
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += len(document_section.content)
        num_tokens = num_tokens_from_string(document_section.content.replace("\n", " ")+"".join(cur), "gpt2")
        if count>2 or num+num_tokens>1500:
            if len(cur)>3 and num+num_tokens>1500 :
                del cur[0]
            break
            
        chosen_sections.append(SEPARATOR  + ": " + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
        count+=1
        num+=num_tokens
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    header = """You are a helpful assistant who answer questions. Answer the question as truthfully as possible the provided context. And if you're unsure of the answer, say "Sorry, I don't know". 答案用中文回答。"\n\nContext:\n"""
    print(header + "".join(chosen_sections) + '\nThis is  the conversation so far.\n'+"".join(cur))
    num_tokens = num_tokens_from_string(header +"".join(chosen_sections) + "".join(cur), "gpt2")
    print(num_tokens)
    return header + "".join(chosen_sections) + "".join(cur)
def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    messages,
    currentme,
    show_prompt: bool = False,
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df,
        currentme
    )
    
    if show_prompt:
        print(len(prompt))
        print(prompt)
    messages.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages = messages
            )

    return response.choices[0].message.content


with open('TEST.pdf', 'rb') as f:
    # 创建一个 PyPDF2 的 PdfFileReader 对象
    pdf_reader = PyPDF2.PdfReader(f)

    # 获取 PDF 文件中的页数
    num_pages = len(pdf_reader.pages)

    # 创建一个空的字符串，用于存储 PDF 文本
    paragraphs = []
    count=0
    tem=''
    # 循环遍历 PDF 文件中的每一页，将其转换为文本并添加到 pdf_text 中
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        page_paragraphs = text.split('\n')
        for paragraph in page_paragraphs:
            if len(tem)<6000:
                tem=tem+paragraph
                tem=str(tem)
            else:
                paragraphs.append(tem)
                tem=''

df = pd.DataFrame({'content': paragraphs})

# 将段落存储为 pandas DataFrame 格式
print(paragraphs)
print(len(paragraphs))
df = pd.DataFrame({'content': paragraphs})

# 打印 DataFrame
print(df.head())
print(f"{len(df)} rows in the data.")


#加载向量
document_embeddings = compute_doc_embeddings(df)
todf = pd.DataFrame.from_dict(document_embeddings)
todf = todf.transpose()
todf.to_csv('embedding-150.csv')

#直接load已有向量
#document_embeddings=load_embeddings('embedding-150.csv')

example_entry = list(document_embeddings.items())[0]

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

f"Context separator contains {separator_len} tokens"
themessage=[]
currentmessage=[]
while True:
    query=input('say')
    currentmessage.append('Q:'+query+'\n')
    an=answer_query_with_context(query, df, document_embeddings,themessage,currentmessage)
    currentmessage.append('A:'+an+'\n')
    num_tokens = num_tokens_from_string(an, "gpt2")