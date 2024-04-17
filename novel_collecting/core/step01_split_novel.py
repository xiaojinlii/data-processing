"""
步骤01
将长篇小说切割成小块
"""
import os
import shutil
import tiktoken
from tqdm import tqdm

from . import settings

enc = tiktoken.get_encoding("cl100k_base")


def split_chapters(raw_text: str):
    chapters = []
    chapter_contents = []

    for line in raw_text.split('\n'):
        head_flag = False
        if line.strip().startswith('第'):
            # 遇到章节标题,将之前章节内容添加到结果列表

            head = line.strip()
            # print(head)
            head = head[:min(10, len(head))]
            if head.find('章', 1) > 0:
                # print(head)
                head_flag = True

        if head_flag:
            if chapter_contents:
                chapters.append('\n'.join(chapter_contents))
                chapter_contents = []
            # 记录当前章节标题
            # chapters.append(line)
        else:
            # 累积章节内容
            chapter_contents.append(line)

    # 添加最后一个章节内容
    if chapter_contents:
        chapters.append('\n'.join(chapter_contents))

    return chapters


def divide_str(s, sep=['\n', '.', '。']):
    mid_len = len(s) // 2  # 中心点位置
    best_sep_pos = len(s) + 1  # 最接近中心点的分隔符位置
    best_sep = None  # 最接近中心点的分隔符
    for curr_sep in sep:
        sep_pos = s.rfind(curr_sep, 0, mid_len)  # 从中心点往左找分隔符
        if sep_pos > 0 and abs(sep_pos - mid_len) < abs(best_sep_pos -
                                                        mid_len):
            best_sep_pos = sep_pos
            best_sep = curr_sep
    if not best_sep:  # 没有找到分隔符
        return s, ''
    return s[:best_sep_pos + 1], s[best_sep_pos + 1:]


def strong_divide(s):
    left, right = divide_str(s)

    if right != '':
        return left, right

    whole_sep = ['\n', '.', '，', '、', ';', ',', '；',
                 '：', '！', '？', '(', ')', '”', '“',
                 '’', '‘', '[', ']', '{', '}', '<', '>',
                 '/', '''\''', '|', '-', '=', '+', '*', '%', \
               '$', '''  # ''', '@', '&', '^', '_', '`', '~',\
                      '·', '…']
    left, right = divide_str(s, sep=whole_sep)

    if right != '':
        return left, right

    mid_len = len(s) // 2
    return s[:mid_len], s[mid_len:]


def split_chunk(chapters, max_token_len):
    chunk_text = []

    for chapter in chapters:

        split_text = chapter.split('\n')

        curr_len = 0
        curr_chunk = ''

        tmp = []

        for line in split_text:
            line_len = len(enc.encode(line))

            if line_len <= max_token_len - 5:
                tmp.append(line)
            else:
                # print('divide line with length = ', line_len)
                path = [line]
                tmp_res = []

                while path:
                    my_str = path.pop()
                    left, right = strong_divide(my_str)

                    len_left = len(enc.encode(left))
                    len_right = len(enc.encode(right))

                    if len_left > max_token_len - 15:
                        path.append(left)
                    else:
                        tmp_res.append(left)

                    if len_right > max_token_len - 15:
                        path.append(right)
                    else:
                        tmp_res.append(right)

                for line in tmp_res:
                    tmp.append(line)

        split_text = tmp

        for line in split_text:
            line_len = len(enc.encode(line))

            if line_len > max_token_len:
                print('warning line_len = ', line_len)

            if curr_len + line_len <= max_token_len:
                curr_chunk += line
                curr_chunk += '\n'
                curr_len += line_len
                curr_len += 1
            else:
                chunk_text.append(curr_chunk)
                curr_chunk = line
                curr_len = line_len

        if curr_chunk:
            chunk_text.append(curr_chunk)

        # break
    return chunk_text


if __name__ == "__main__":
    raw_novel_text = open(settings.raw_novel_path, encoding='utf-8').read()
    novel_chapters = split_chapters(raw_novel_text)   # 将整篇小说按章节分割
    chunks = split_chunk(novel_chapters, 1500)  # 将所有章节分割成chunk块，每个chunk块上限1500tokens

    save_chunk_path = os.path.join(settings.novel_temp_path, "raws")
    shutil.rmtree(save_chunk_path, True)
    os.makedirs(save_chunk_path, exist_ok=True)

    for i in tqdm(range(len(chunks))):
        raw_text_save_name = os.path.join(save_chunk_path, f"{i}_raw.txt")
        with open(raw_text_save_name, 'w', encoding='utf-8') as f:
            f.write(chunks[i])
