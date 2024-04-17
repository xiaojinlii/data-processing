# novel collecting
小说角色台词采集

## 安装
```
pip install -r novel_collecting\requirements.txt
```

## 配置
1. 在根目录下的`settings.py`中配置模型接口
2. 在`novel_collecting\settings.py`中配置小说相关内容

## 使用步骤
1. 将长篇小说切割成chunk块
    ```
    python -m novel_collecting.core.step01_split_novel
    ```
2. 抽取每个chunk块的对话
    ```
    python -m novel_collecting.core.step02_generate_dialogue
    ```
3. 为每个chunk块生成总结
    ```
    python -m novel_collecting.core.step03_generate_summarize
    ```
4. 将dialogue和summarize合并到jsonl文件
    ```
    python -m novel_collecting.core.step04_export_jsonl
    ```
5. 抽取角色对话
   ```
   python -m novel_collecting.core.step05_extract_role
   ```
6. 生成指定角色的人物形象设定
   ```
   python -m novel_collecting.core.step06_generate_prompt
   ```
   在/temp/characters/角色名路径下创建system_prompt.txt文件，并将生成的人物形象设定写入
7. 生成台词jsonl文件
   ```
   python -m novel_collecting.core.step07_embeddings
   ```
   