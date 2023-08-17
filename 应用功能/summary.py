import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI

os.environ["OPENAI_API_KEY"] = "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"
# 导入文本
loader = UnstructuredFileLoader("test.txt")
# 将文本转成 Document 对象
document = loader.load()
print(f'documents:{len(document)}')

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 0
)

# 切分文本
split_documents = text_splitter.split_documents(document)
print(f'documents:{len(split_documents)}')

# 加载 llm 模型
llm = OpenAI(model_name="text-davinci-003", max_tokens=1500)

# 创建总结链，进行长文总结时，chain_type可为refine，map_reduce
chain = load_summarize_chain(llm, chain_type="refine", verbose=True)

# 执行总结链
chain.run(split_documents)