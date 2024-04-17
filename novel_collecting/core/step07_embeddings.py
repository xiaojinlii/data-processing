"""
步骤07
生成台词jsonl文件
"""
import json
import os.path

from novel_collecting.core.utils import get_embeddings, package_role
from . import settings


def write_jsonl(datas, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for data in datas:
            json_str = json.dumps(data, ensure_ascii=False)
            f.write(json_str + "\n")


if __name__ == "__main__":
    role_name_en = "zhongling"
    texts_path = os.path.join(settings.novel_temp_path, "characters", role_name_en, "texts")
    save_path = os.path.join(settings.novel_temp_path, "characters", role_name_en, f"{role_name_en}.jsonl")
    system_prompt_path = os.path.join(settings.novel_temp_path, "characters", role_name_en, "system_prompt.txt")

    with open(system_prompt_path, 'r', encoding='utf-8') as file:
        system_prompt = file.read()

    embedding = get_embeddings()
    datas = package_role(system_prompt, texts_path, embedding)
    write_jsonl(datas, save_path)
    print(save_path)
