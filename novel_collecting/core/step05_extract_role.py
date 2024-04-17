"""
步骤05
抽取角色对话
"""
import json
import os
import zipfile
from collections import Counter
import tiktoken

from . import settings

enc = tiktoken.get_encoding("cl100k_base")


def get_data(reorganized_path):
    data_in_chunk = []
    with open(reorganized_path, encoding='utf8') as f:
        for line in f:
            data_in_chunk.append(json.loads(line))

    data = []

    for chunk in data_in_chunk:
        for d in chunk:
            data.append(d)

    # print(len(data))
    for i, d in enumerate(data):
        if d['role'] == 'scene':
            first_scene_id = i
            break
    # print(first_scene_id)
    return data, first_scene_id


def get_roles(data):
    role_counts = Counter()
    for line in data:
        role = line['role']
        role_counts[role] += 1

    common_roles = role_counts.most_common(30)

    status = 0

    role_name = []

    for role, count in common_roles:
        status = status + 1
        if role != 'scene':
            role_name.append(role)
        if status % 3 == 0:
            print(role, count)
        else:
            print(role, count, end=' ')
    sorted_roles = sorted(common_roles, key=lambda x: x[1], reverse=True)
    sorted_roles_clear = [role[0] for role in sorted_roles if role[0] != "scene"]
    return sorted_roles_clear


# 联系角色和旁白
def output_scene_chat_id(data, target_role_single):
    chat_ids = []

    # 先寻找所有出现角色的节点
    for i, d in enumerate(data):
        if d['role'] == target_role_single:
            chat_ids.append(i)

    previous_scene_ids = []

    # 对于每一个chat_ids，向前寻找scene的节点
    for chat_id in chat_ids:
        ans = first_scene_id
        for j in range(chat_id, first_scene_id, -1):
            if data[j]['role'] == 'scene':
                ans = j
                break
        previous_scene_ids.append(ans)
    return chat_ids, previous_scene_ids


"""
分块，决定texts组织内容
chat_ids是一个list of int。代表chat出现的顺序，是个严格升序的序列

我希望找到chat_ids中，所有的元素之间间隔不大于max_find_lines = 10连续子串，用list of list of int的形式，保存到chat_ids_in_chunk

例子输入 1,2,4,100,110,120,555 例子输出 [1,2,4],[100,110,120],[555]

请用python为我实现"""


def divide_chats2chunks(chat_ids, previous_scene_ids, cur_role_name):
    chat_ids_in_chunk = []
    current_chunk = []

    for chat_id in chat_ids:
        if not current_chunk:
            current_chunk.append(chat_id)
            continue

        if chat_id - current_chunk[-1] <= max_find_lines:
            current_chunk.append(chat_id)
        else:
            chat_ids_in_chunk.append(current_chunk)
            current_chunk = [chat_id]

    if current_chunk:
        chat_ids_in_chunk.append(current_chunk)

    # print(chat_ids_in_chunk[0])

    chat_id2previous_scene_id = {}

    for previous, chat_id in zip(previous_scene_ids, chat_ids):
        chat_id2previous_scene_id[chat_id] = previous
        if previous > 0:
            if data[previous - 1]['role'] != cur_role_name:
                chat_id2previous_scene_id[chat_id] -= 1
    return chat_ids_in_chunk, chat_id2previous_scene_id


"""
组织texts
计算一下每一句所花的token数量
"""


def count_token(my_str):
    return len(enc.encode(my_str))


def data2str(data):
    role = data['role']
    if role in ['旁白', '', 'scene', 'Scene', 'narrator', 'Narrator']:
        return 'scene:' + data['text']
    else:
        return role + ':「' + data['text'] + '」'


# 我们现在需要把这东西变成最终的texts文本
def id2texts(data, chat_ids_in_chunk, chat_id2previous_scene_id):
    line_token = [count_token(data2str(d)) for d in data]
    from ast import Break
    final_chunks = []

    print_count = 0

    appended_key = []

    for chunk in chat_ids_in_chunk:
        N = len(chunk)

        current_i = 0

        while current_i < N - 1:

            consider_chat_id = chunk[current_i]

            previous_scene_id = chat_id2previous_scene_id[consider_chat_id]

            # 保底
            withdraw_start = previous_scene_id
            withdraw_end = consider_chat_id

            current_count = sum(line_token[previous_scene_id:consider_chat_id + 1])
            while current_count < max_token_num and current_i < N - 1:
                consider_end = chunk[current_i + 1]
                consider_count = sum(line_token[previous_scene_id:consider_end + 1])
                if consider_count < max_token_num:
                    current_count = consider_count
                    withdraw_start = previous_scene_id
                    withdraw_end = consider_end
                    current_i += 1
                else:
                    break

            # print_count += 1

            # print(withdraw_start, end = ' ')
            # if print_count % 5 == 0:
            #     print()

            if withdraw_end + 1 not in appended_key:
                appended_key.append(withdraw_end + 1)
                chunk_str = ''
                for i in range(withdraw_start, withdraw_end + 1):
                    chunk_str += data2str(data[i]) + '\n'

                final_chunks.append(chunk_str)

            current_i += 1
    return appended_key, final_chunks


# 保存成不同的形式
def save_chunk2zip(temp_path, role_name_en, final_chunks):
    save_path = os.path.join(temp_path, role_name_en)
    text_path = os.path.join(save_path, "texts")
    os.makedirs(text_path, exist_ok=True)

    for i in range(0, len(final_chunks)):
        my_str = final_chunks[i]
        with open(text_path + f'/text_{i}.txt', 'w', encoding='utf-8') as f:
            f.write(my_str)

    zip_path = os.path.join(save_path, f"texts.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for filename in os.listdir(text_path):
            zipf.write(text_path + "/" + filename)

    # print('Zipped folder saved to', zip_path)


if __name__ == "__main__":
    # 支持跨越多少行寻找目标角色，也即控制段内行间距不超过该值
    max_find_lines = 10
    max_token_num = 500

    reorganized_path = os.path.join(settings.novel_temp_path, "reorganized/reorganized.jsonl")

    data, first_scene_id = get_data(reorganized_path)
    all_roles = get_roles(data)

    # 要抽取的角色名
    role_extract = ["阿紫", "阿朱", "王语嫣", "慕容复", "钟灵"]
    role_extract_en = {
        "段誉": "duanyu",
        "乔峰": "qiaofeng",
        "虚竹": "xuzhu",
        "阿紫": "azi",
        "阿朱": "azhu",
        "王语嫣": "wangyuyan",
        "慕容复": "murongfu",
        "钟灵": "zhongling",
    }

    for role_name in role_extract:
        if role_name in all_roles:
            chat_ids, previous_scene_ids = output_scene_chat_id(data, role_name)
            chat_ids_in_chunk, chat_id2previous_scene_id = divide_chats2chunks(chat_ids, previous_scene_ids, role_name)
            appended_key, final_chunks = id2texts(data, chat_ids_in_chunk, chat_id2previous_scene_id)
            role_en = role_extract_en[role_name]
            save_chunk2zip(os.path.join(settings.novel_temp_path, "characters"), role_en, final_chunks)
        else:
            print(f"Error: {role_name} not exist.")
