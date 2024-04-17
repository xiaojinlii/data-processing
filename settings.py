

# openai api形式接口
MODEL_NAME = "gpt-3.5-turbo"
API_BASE = "https://api.openai.com/v1"
API_KEY = "empty"


# embeddings配置
# 是否使用本地embeddings，默认使用本地embeddings
LOCAL_EMBEDDINGS = True
# 本地embeddings模型路径
EMBEDDINGS_MODEL_PATH = r"E:\WorkSpace\LLMWorkSpace\Models\Embedding\bge-large-zh-v1.5"
# 远程embeddings接口, 使用 https://github.com/xiaojinlii/fastllm 远程部署
EMBEDDINGS_API_BASE = "http://10.12.25.5:21021/embeddings"
