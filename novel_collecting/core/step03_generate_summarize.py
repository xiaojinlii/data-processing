"""
步骤03
为每个chunk块生成总结
注意：如果某个chunk块存在已总结的文件，为了节省tokens以及异常中断导致未处理完全部chunk，默认不处理
"""
import os
import time

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from tqdm import tqdm

from . import settings
from .utils import read_chunks, get_chat_model

system_prompt = """
Summarize the key points of the following text in a concise way, using bullet points.
"""

q_example = """###
Text:
洪七公、周伯通、郭靖、黄蓉四人乘了小船，向西驶往陆地。郭靖坐在船尾扳桨，黄蓉不住向周伯通详问骑鲨游海之事，周伯通兴起，当场就要设法捕捉鲨鱼，与黄蓉大玩一场。
郭靖见师父脸色不对，问道：“你老人家觉得怎样”洪七公不答，气喘连连，声息粗重。他被欧阳锋以“透骨打穴法”点中之后，穴道虽已解开，内伤却又加深了一层。黄蓉喂他服了几颗九花玉露丸，痛楚稍减，气喘仍是甚急。
老顽童不顾别人死活，仍是嚷着要下海捉鱼，黄蓉却已知不妥，向他连使眼色，要他安安静静的，别吵得洪七公心烦。周伯通并不理会，只闹个不休。黄蓉皱眉道：“你要捉鲨鱼，又没饵引得鱼来，吵些甚么”

Summarize in BULLET POINTS form:
"""

a_example = """
- 洪七公等四人乘船西行,洪七公因受内伤加重而气喘不止
- 周伯通要捉鲨鱼玩,被黄蓉阻止以免掀翻小船
"""


def generate_summarize(i, raw, save_path, override=False):
    file_path = os.path.join(save_path, f"{i}_sum.txt")
    if os.path.exists(file_path) and override is False:
        return

    time.sleep(15)
    messages = [SystemMessage(content=system_prompt),
                HumanMessage(content=q_example),
                AIMessage(content=a_example)]

    new_input = f"###\nText:\n{raw}\nSummarize in BULLET POINTS form:"

    messages.append(HumanMessage(content=new_input))

    summarize_response = llm.invoke(messages).content

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(summarize_response)


def generate_all_summarize(chunks, save_path):
    for i in tqdm(range(len(chunks))):
        try:
            generate_summarize(i, chunks[i], save_path)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    save_chunk_path = os.path.join(settings.novel_temp_path, "raws")
    save_summarize_path = os.path.join(settings.novel_temp_path, "summarizes")
    os.makedirs(save_summarize_path, exist_ok=True)

    llm = get_chat_model()

    chunks = read_chunks(save_chunk_path)
    # generate_summarize(0, chunks[0], save_summarize_path)
    generate_all_summarize(chunks, save_summarize_path)
