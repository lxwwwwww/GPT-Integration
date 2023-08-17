import openai
import time

# 设置 openai API 密钥
openai.api_key = "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"
response = openai.Image.create(
  prompt="a white siamese cat",
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']
print(image_url)