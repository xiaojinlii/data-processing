"""
步骤04
将dialogue和summarize合并到jsonl文件
"""
import shutil

import numpy as np
import os
import copy
import json
from tqdm import tqdm

from . import settings


# 给定长文本raw_text。
# 使用换行符\n或者。 来对这个字符串进行切割，忽略掉strip之后是空的子字符串
# 将每一段话的起点位置存储在一个list of int , starts中
# 将每一段话的结束位置存储在一个list of int , ends中
# 并且将每一个子字符串的存储在一个list of str, lines中
def divide_raw2lines(raw_text):
    previous_str = ''
    starts = []
    ends = []
    lines = []
    for i in range(len(raw_text)):
        previous_str += raw_text[i]
        if raw_text[i] in ('\n', '。'):
            strip_str = previous_str.strip(' "“”\r\n')
            if len(strip_str) > 0:
                lines.append(strip_str)
                starts.append(i - len(strip_str))
                ends.append(i)
            previous_str = ''
        else:
            pass
    return lines, starts, ends


# 已知if '\u4e00' <= char <= '\u9fa5': 可以判断一个char是否是中文字
# 我希望实现一个函数，这个函数的输入是两个list of string, 长度为M的query 和 长度为N的datas
# 输出是一个M*N的numpy float数组 recalls
# 先计算freqs[m][n] 表示query的第m句中的每一个中文字，是否在datas[n]中是否出现，如果出现，则freqs[m][n]加一
# 然后计算recalls[m][n]是freqs[m][n]除掉 query[m]中所有中文字的个数
def compute_char_recall(query, datas):
    M = len(query)
    N = len(datas)

    freqs = np.zeros((M, N), dtype=int)

    for m in range(M):
        q_chars = set()
        for char in query[m]:
            if '\u4e00' <= char <= '\u9fa5':
                q_chars.add(char)

        for n in range(N):
            for char in q_chars:
                if char in datas[n]:
                    freqs[m][n] += 1

    query_chars_count = [len(set(char for char in sent if '\u4e00' <= char <= '\u9fa5'))
                         for sent in query]

    recalls = freqs / np.array(query_chars_count)[:, None]

    return recalls


# 给定长文本raw_text。
# 使用换行符\n或者。 来对这个字符串进行切割，忽略掉strip之后是空的子字符串
# 将每一段话的起点位置存储在一个list of int , starts中
# 将每一段话的结束位置存储在一个list of int , ends中
# 并且将每一个子字符串的存储在一个list of str, lines中
def divide_raw2lines(raw_text):
    previous_str = ''
    starts = []
    ends = []
    lines = []
    for i in range(len(raw_text)):
        previous_str += raw_text[i]
        if raw_text[i] in ('\n', '。'):
            strip_str = previous_str.strip(' "“”\r\n')
            if len(strip_str) > 0:
                lines.append(strip_str)
                starts.append(i - len(strip_str))
                ends.append(i)
            previous_str = ''
        else:
            pass
    return lines, starts, ends


# 已知if '\u4e00' <= char <= '\u9fa5': 可以判断一个char是否是中文字
# 我希望实现一个函数，这个函数的输入是两个list of string, 长度为M的query 和 长度为N的datas
# 输出是一个M*N的numpy float数组 recalls
# 先计算freqs[m][n] 表示query的第m句中的每一个中文字，是否在datas[n]中是否出现，如果出现，则freqs[m][n]加一
# 然后计算recalls[m][n]是freqs[m][n]除掉 query[m]中所有中文字的个数
def compute_char_recall(query, datas):
    M = len(query)
    N = len(datas)

    freqs = np.zeros((M, N), dtype=int)

    for m in range(M):
        q_chars = set()
        for char in query[m]:
            if '\u4e00' <= char <= '\u9fa5':
                q_chars.add(char)

        for n in range(N):
            for char in q_chars:
                if char in datas[n]:
                    freqs[m][n] += 1

    query_chars_count = [len(set(char for char in sent if '\u4e00' <= char <= '\u9fa5'))
                         for sent in query]

    recalls = freqs / np.array(query_chars_count)[:, None]

    return recalls


def summary2line(chunk_sum, lines):
    s = compute_char_recall(chunk_sum, lines)

    color_map = {}
    ans_Q = {}

    ans_div = {}

    flags = {}

    M = len(chunk_sum)
    N = len(lines)

    for n in range(0, N):
        if n == 0:
            ans_Q[(0, 0)] = s[0, 0]
            ans_div[(0, 0)] = []
        else:
            ans_Q[(0, n)] = ans_Q[(0, n - 1)] + s[0, n]
            ans_div[(0, n)] = []

    for m in range(1, M):
        ans_Q[(m, m)] = ans_Q[(m - 1, m - 1)] + s[m, m]
        ans_div[(m, m)] = ans_div[(m - 1, m - 1)].copy()
        ans_div[(m, m)].append(m)

    def find_Q(m, n):
        # print(m,n)

        if m < 0 or n < 0:
            print('error out bound', m, ' ', n)
            return 0, []

        if (m, n) in ans_Q.keys():
            return ans_Q[(m, n)], ans_div[(m, n)]

        if (m, n) in color_map.keys():
            print('error repeated quest ', m, ' ', n)
            return 0, []
        else:
            color_map[(m, n)] = 1

        current_div = []

        left, left_div = find_Q(m, n - 1)
        right, right_div = find_Q(m - 1, n - 1)

        if left > right:
            ans = left + s[m][n]
            flags[(m, n)] = False
            current_div = left_div

        else:
            ans = right + s[m][n]
            flags[(m, n)] = True
            current_div = right_div.copy()
            current_div.append(n - 1)

        # ans = max(  , ) + s[m][n]

        ans_Q[(m, n)] = ans
        ans_div[(m, n)] = current_div.copy()

        return ans, current_div

    # print(find_Q(0,5))
    # print(find_Q(M-1,N-1))

    score, divs = find_Q(M - 1, N - 1)
    divs.append(N - 1)

    return score, divs


def dialogue2line(dia_texts, lines):
    """
    s_dialogue 存储了一个M*N的np array
    我现在希望实现一个python程序，能够找到一个长度为M的顺序子序列 a_0, ... , a_m-1
    使得s_dialogue[ i ][ a_i ] 之和最大
    输出a_0, ... , a_{m-1}的值
    用动态规划算法，python为我实现
    """
    s_dialogue = compute_char_recall(dia_texts, lines)

    M, N = s_dialogue.shape
    if M == 0 or N == 0:
        return []
    dp = np.zeros((M, N))
    dp[0] = s_dialogue[0]
    prev_indices = np.zeros((M, N), dtype=int)
    for i in range(1, M):
        for j in range(N):
            max_prev_index = np.argmax(dp[i - 1])
            dp[i][j] = dp[i - 1][max_prev_index] + s_dialogue[i][j]
            prev_indices[i][j] = max_prev_index

    max_end_index = np.argmax(dp[-1])
    sequence = []
    for i in range(M - 1, -1, -1):
        sequence.append(max_end_index)
        max_end_index = prev_indices[i][max_end_index]
    sequence.reverse()

    return sequence


def jsonl_sorted(chunk_sum, divs, dia_texts, seq):
    combined_data = []
    combined_text = ""
    for index in sorted(seq + divs):
        # print(index)
        if index in seq:
            combined_data.append({
                "role": dialogues[seq.index(index)]["role"],
                'text': dialogues[seq.index(index)]["dialogue"],
                'if_scene': False
            })
            combined_text = combined_text + dialogues[seq.index(index)]["role"] + ":" + dialogues[seq.index(index)][
                "dialogue"] + "\n"
            seq[seq.index(index)] = -1
        if index in divs:
            combined_data.append({
                "role": "scene",
                'text': chunk_sum[divs.index(index)],
                'if_scene': True
            })
            combined_text = combined_text + "scene" + ":" + chunk_sum[divs.index(index)] + "\n"
            divs[divs.index(index)] = -1

    return combined_data, combined_text


if __name__ == "__main__":
    raws_path = os.path.join(settings.novel_temp_path, "raws")
    reorganized_path = os.path.join(settings.novel_temp_path, "reorganized")

    shutil.rmtree(reorganized_path, True)
    os.makedirs(reorganized_path, exist_ok=True)
    save_jsonl_path = os.path.join(reorganized_path, "reorganized.jsonl")
    save_txt_path = os.path.join(reorganized_path, "reorganized.txt")

    chunk_text = [""] * (sum([1 for file_name in os.listdir(raws_path) if file_name.endswith("_raw.txt")]))
    # 遍历文件夹中的文件
    for file_name in os.listdir(raws_path):
        if file_name.endswith("_raw.txt"):
            file_path = os.path.join(raws_path, file_name)

            i = int(file_name.split('_')[0])

            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            chunk_text[i] = text

    final_jsonl = []
    final_txt = ""

    for i in tqdm(range(0, len(chunk_text)), desc="Processing", total=len(chunk_text) - 1, unit="item"):
        try:
            # print(f"index:{i}")
            raw_text = chunk_text[i]

            dialoge_file = os.path.join(settings.novel_temp_path, f"dialogues/{i}_dialogue.txt")
            summarzie_file = os.path.join(settings.novel_temp_path, f"summarizes/{i}_sum.txt")

            chunk_sum = []
            unique_chunk_sum = []
            if os.path.exists(summarzie_file):
                with open(summarzie_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip().startswith('-'):
                            chunk_sum.append(line.strip()[1:].strip())
            if os.path.exists(dialoge_file):
                with open(dialoge_file, encoding='utf-8') as f:
                    dialogues = []
                    for line in f:
                        dialogue = json.loads(line)
                        dialogues.append(dialogue)

            unique_dialogue = []
            for item in dialogues:
                if item not in unique_dialogue:
                    unique_dialogue.append(item)
            dia_texts = [data['dialogue'] for data in unique_dialogue]
            for item in chunk_sum:
                if item not in unique_chunk_sum:
                    unique_chunk_sum.append(item)

            chunk_sum = unique_chunk_sum
            dialogues = unique_dialogue
            # print(f"sum:{chunk_sum}")
            # print(f"dialogues:{dialogues}")
            lines, starts, ends = divide_raw2lines(raw_text)
            # print(f"lines:{lines}")
            score, divs = summary2line(chunk_sum, lines)  # summary匹配
            # print(f"score:{score}  divs:{divs}")
            seq = dialogue2line(dia_texts, lines)  # 对话匹配
            # print(f"seq:{seq}")
            combined_data, combined_text = jsonl_sorted(chunk_sum, divs.copy(), dia_texts, seq.copy())
            # print(f"combined_data:{combined_data}")
            # print(f"combined_text:{combined_text}")
            # 如果需要保存每个chunk的，在此处保存
            final_jsonl.append(combined_data)
            final_txt = final_txt + combined_text + "\n"
        except:
            print("第" + str(i) + "个chunk出错")
            pass
    with open(save_jsonl_path, "w", encoding="utf-8") as file:
        # 遍历数据列表中的每个字典
        for record in final_jsonl:
            # 将字典转换为JSON格式的字符串
            json_record = json.dumps(record, ensure_ascii=False)
            # 将转换后的JSON字符串写入文件，并添加换行符
            file.write(json_record + "\n")
    with open(save_txt_path, "w", encoding="utf-8") as file:
        file.write(final_txt)
