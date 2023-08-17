import openai
import time

# 设置 openai API 密钥
openai.api_key = "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"

# 定义问答函数
def ask(current_messages,question):
    messages=[]
    messages.append({"role": "user", "content": 'The previous conversation :\n'+current_messages+'\nThis is the current question:'+question})
    # 调用 OpenAI API 进行文本生成
    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages = messages
            )

    # 获取生成的答案
    answer = response.choices[0].message.content

    return answer

# 进行对话
current_messages = ''
while True:
    # 读取用户输入的问题
    question = input("你: ")

    # 如果用户输入 "退出"，则退出对话
    if question == "退出":
        print("再见！")
        break
    # 向 GPT-3.5 提问，并获取回答
    answer = ask(current_messages,question)

    # 输出回答
    print("GPT-3.5:", answer)

    # 将上下文信息更新为包括最新的问题和回答
    current_messages += f"\nQ: {question}\nA: {answer}"

    # 延迟 1 秒，避免过快地发送请求
    time.sleep(1)