import base64
import struct

import os
from tqdm import tqdm
from utils.models import get_chat_model, get_embeddings


def read_chunks(path):
    chunks = [""] * (sum([1 for file_name in os.listdir(path) if file_name.endswith("_raw.txt")]))

    for file_name in os.listdir(path):
        if file_name.endswith("_raw.txt"):
            file_path = os.path.join(path, file_name)
            i = int(file_name.split('_')[0])
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            chunks[i] = text

    return chunks


def float_array_to_base64(float_arr):
    byte_array = b''

    for f in float_arr:
        # 将每个浮点数打包为4字节
        num_bytes = struct.pack('!f', f)
        byte_array += num_bytes

    # 将字节数组进行base64编码
    base64_data = base64.b64encode(byte_array)

    return base64_data.decode('utf-8')


def package_role(system_prompt, texts_path, embedding):
    datas = []

    # 暂时只有一种embedding 'luotuo_openai'
    embed_name = 'luotuo_openai'

    datas.append({'text': system_prompt, embed_name: 'system_prompt'})
    datas.append({'text': 'Reserve Config Setting Here', embed_name: 'config'})

    # debug_count = 3

    # for file in os.listdir(texts_path):

    files = os.listdir(texts_path)

    for i in tqdm(range(len(files))):
        file = files[i]
        # if file name end with txt
        if file.endswith(".txt"):
            file_path = os.path.join(texts_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                current_str = f.read()
                current_vec = embedding.embed_documents([current_str])
                encode_vec = float_array_to_base64(current_vec[0])
                datas.append({'text': current_str, embed_name: encode_vec})

                # debug_count -= 1
                # if debug_count == 0:
                #     break
    return datas
