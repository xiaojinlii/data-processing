"""
步骤06
生成指定角色的人物形象设定
"""
import codecs
import os
import random

import tiktoken
from langchain_core.messages import SystemMessage, HumanMessage

from . import settings
from .utils import get_chat_model

enc = tiktoken.get_encoding("cl100k_base")


def get_chunks(role_text_path, max_predict_token=2500):
    role_texts = []

    for filename in os.listdir(role_text_path):
        if filename.endswith('.txt'):
            with codecs.open(os.path.join(role_text_path, filename), 'r', 'utf-8') as f:
                role_texts.append(f.read())

    random.shuffle(role_texts)

    role_chunk = []
    chunk = ''
    current_len = 0
    for text in role_texts:
        len_text = len(enc.encode(text))
        if current_len + len_text <= max_predict_token:
            chunk += '\n\n' + text
            current_len += (2 + len_text)
        else:
            role_chunk.append(chunk)
            chunk = text
            current_len = len_text

    # for last chunk add more texts from the head of role_texts
    if chunk:
        for text in role_texts:
            len_text = len(enc.encode(text))
            if current_len + len_text <= max_predict_token:
                chunk += '\n\n' + text
                current_len += (2 + len_text)
            else:
                break
        role_chunk.append(chunk)

    # for chunk in role_chunk:
    #     print(len(enc.encode(chunk)), end=' ')

    return role_chunk


def generate_prompt(role_chunk, prefix_prompt, smart_system_prompt):
    responses = []
    count = 0
    n = 3
    llm = get_chat_model()

    for chunk in role_chunk:
        print('index = ', count)
        whole_message = prefix_prompt + "```\n" + chunk + "\n```"
        messages = [
            SystemMessage(content=smart_system_prompt),
            HumanMessage(content=whole_message),
        ]

        if count <= 1:
            pass
            # print(whole_message)
        else:
            response = llm.invoke(messages)
            responses.append(response)

        count = count + 1
        if count > n + 1:
            break

    for response in responses:
        print(response.content)
        print('\n----------\n')


if __name__ == "__main__":
    # 要处理的角色信息
    role_name_en = "zhongling"
    role_name = '钟灵'
    world_name = '天龙八部'

    role_text_path = os.path.join(settings.novel_temp_path, "characters", role_name_en, "texts")
    role_chunk = get_chunks(role_text_path, 2500)


    prefix_prompt = f'''
    你在分析小说{world_name}中的角色{role_name}
    结合小说{world_name}中的内容，以及下文中角色{role_name}的对话
    判断{role_name}的人物设定、人物特点以及语言风格

    {role_name}的对话:
    '''

    smart_system_prompt = "You are ChatGPT, a large language model trained by OpenAI."

    generate_prompt(role_chunk, prefix_prompt, smart_system_prompt)
