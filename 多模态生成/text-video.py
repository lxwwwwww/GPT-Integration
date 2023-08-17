import deep_dream_video

# 设置输入文本和输出视频的文件路径
input_text_path = "input.txt"
output_video_path = "output.mp4"

# 设置 DeepDreamVideo 的参数
params = {
    "duration": 10,  # 视频时长
    "fps": 30,  # 帧率
    "resolution": (640, 480),  # 分辨率
    "text_color": "white",  # 文字颜色
    "background_color": "black",  # 背景颜色
    "font_file": "arial.ttf",  # 字体文件路径
    "font_size": 36,  # 字体大小
    "model_file": "model.h5",  # 模型文件路径
}

# 读取输入文本
with open(input_text_path, "r") as f:
    input_text = f.read()

# 调用 DeepDreamVideo 生成视频
deep_dream_video.generate_video(input_text, output_video_path, **params)