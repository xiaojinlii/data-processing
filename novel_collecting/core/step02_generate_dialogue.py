"""
步骤02
抽取每个chunk块的对话
注意：如果某个chunk块存在已抽取的对话文件，为了节省tokens以及异常中断导致未处理完全部chunk，默认不处理
"""
import json
import os
import time

from kor import create_extraction_chain
from kor.nodes import Object, Text
from tqdm import tqdm

from . import settings
from .utils import read_chunks, get_chat_model

schema = Object(
    id="script",
    description="Extract Dialogue in order From Novel, ignore the non-dialogue parts",
    attributes=[
        Text(
            id="role",
            description="The character who is speaking, use context to predict the name of the role.",
        ),
        Text(
            id="dialogue",
            description="The dialogue spoken by the characters in the sentence",
        ),
    ],
    examples=[
        (
            '''``村民中走出一个二十来岁的人汉，说道：“张先生，你可是从北方来吗”

    张十五见他身材魁梧，浓眉大眼，便道：“正是。”那大汉道：“小弟作东，请先生去饮上三杯如何”张十五大喜，说道：“素不相识，怎敢叨扰”

    那大汉笑道：“喝上三怀，那便相识了。我姓郭，名叫郭啸天。”``''',
            [
                {"role": "郭啸天", "dialogue": "张先生，你可是从北方来吗"},
                {"role": "张十五", "dialogue": "正是。"},
                {"role": "郭啸天", "dialogue": "小弟作东，请先生去饮上三杯如何"},
                {"role": "张十五", "dialogue": "素不相识，怎敢叨扰"},
                {"role": "郭啸天", "dialogue": "喝上三怀，那便相识了。我姓郭，名叫郭啸天。"}
            ],
        )
    ],
    many=True,
)


def generate_dialogue(i, raw, save_path, override=False):
    file_path = os.path.join(save_path, f"{i}_dialogue.txt")
    if os.path.exists(file_path) and override is False:
        return

    time.sleep(15) # Rate limit reached for gpt-3.5-turbo in organization org-eW0rw1VKbbcEXjXTM7gKL7Gv on requests per min (RPM): Limit 3, Used 3, Requested 1.
    query_text = f"``{raw}``"
    dialogue_response = chain.run(query_text)["data"]

    with open(file_path, 'w', encoding='utf-8') as f:
        if 'script' not in dialogue_response:
            raise f'Error: response does not contain key "script". i:{i}'
        else:
            for chat in dialogue_response['script']:
                json_str = json.dumps(chat, ensure_ascii=False)
                f.write(json_str + "\n")


def generate_all_dialogue(chunks, save_path):
    for i in tqdm(range(len(chunks))):
        try:
            generate_dialogue(i, chunks[i], save_path)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    save_chunk_path = os.path.join(settings.novel_temp_path, "raws")
    save_dialogue_path = os.path.join(settings.novel_temp_path, "dialogues")
    os.makedirs(save_dialogue_path, exist_ok=True)

    llm = get_chat_model()
    chain = create_extraction_chain(llm, schema)

    chunks = read_chunks(save_chunk_path)
    # generate_dialogue(0, chunks[0], save_dialogue_path)
    generate_all_dialogue(chunks, save_dialogue_path)
