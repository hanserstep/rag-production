"""LLM 调用层（DeepSeek）。

几个刻意的设计：

1. Prompt 里不写"不要编造"。这类否定指令对降低幻觉几乎无效，
   真正有效的是检索侧的相关性阈值（见 reranker.py）。Prompt 只负责
   规定输出格式和引用方式。

2. 上下文按 [来源N] 编号注入，要求模型在句末标注 [来源N]。
   这样答案里的每句话都能被追溯，用户能核对，评测时也能算
   faithfulness。

3. HyDE 单独一个方法，不默认开启。它对开放域问答有效，对术语密集
   的技术文档经常帮倒忙（生成的假设答案引入原文没有的表述），
   必须让调用方显式选择。
"""

import os
from typing import Dict, Generator, List, Tuple

from openai import OpenAI

SYSTEM_PROMPT = """你是企业知识库的问答助手。你的回答必须严格基于给定的参考资料。

规则：
1. 只使用参考资料中的信息作答，不使用你的先验知识补充事实。
2. 每个论断后面用 [来源N] 标注依据，N 对应参考资料编号。
3. 如果参考资料不足以回答问题，直接说明"参考资料中没有相关内容"，不要推测。
4. 回答用中文，简洁准确，涉及步骤时用有序列表。"""

HYDE_PROMPT = """请针对下面的问题，写一段 100 字以内的技术性回答。

这段文字会被用来检索相关文档，所以请使用该领域的专业术语和可能的表述方式，
像技术文档那样写，不要用"我不确定"之类的措辞。

问题：{question}"""


class DeepSeekClient:

    def __init__(self, api_key: str = "", base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat", timeout: float = 60.0):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=base_url, timeout=timeout)

    # ---- RAG ----
    def build_rag_messages(self, question: str, context: str) -> List[Dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"},
        ]

    def complete(self, messages: List[Dict]) -> Tuple[str, Dict]:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.2,
        )
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
        return resp.choices[0].message.content or "", usage

    def stream(self, messages: List[Dict]) -> Generator[str, None, None]:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.2, stream=True,
        )
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ---- HyDE ----
    def generate_hyde(self, question: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": HYDE_PROMPT.format(question=question)}],
                max_tokens=200, temperature=0.3,
            )
            return (resp.choices[0].message.content or question).strip()
        except Exception:
            # HyDE 是增强项，失败不能影响主链路
            return question

    # ---- 成本 ----
    def estimate_cost(self, usage: Dict, cfg) -> float:
        """按 token 单价估算单次成本（元）。"""
        return round(
            usage.get("prompt_tokens", 0) / 1000 * cfg.price_prompt_per_1k
            + usage.get("completion_tokens", 0) / 1000 * cfg.price_completion_per_1k,
            6,
        )
