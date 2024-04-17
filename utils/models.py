from langchain_openai import ChatOpenAI

import settings


def get_chat_model(temperature: int = 0.7) -> ChatOpenAI:
    chat_model = ChatOpenAI(
        model_name=settings.MODEL_NAME,
        openai_api_key=settings.API_KEY,
        openai_api_base=settings.API_BASE,
        temperature=temperature,
    )
    return chat_model


def get_embeddings():
    if settings.LOCAL_EMBEDDINGS:
        # pip install torch sentence-transformers
        import torch
        from langchain_community.embeddings import HuggingFaceEmbeddings

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDINGS_MODEL_PATH, model_kwargs={'device': device})
    else:
        from .langchain_fastllm import FastEmbeddings
        embeddings = FastEmbeddings(settings.EMBEDDINGS_API_BASE)
    return embeddings


if __name__ == "__main__":
    model = get_chat_model()
    res = model.invoke("你是谁")
    print(res)
