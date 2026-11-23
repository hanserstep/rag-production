"""
文档解析 & 分块模块
==================
面试点：为什么 chunk_size=500, overlap=50？

chunk_size 太小（100）：语义碎片化，检索不准
chunk_size 太大（2000）：单个chunk信息太杂，且可能超过LLM上下文窗口
最佳实践：500-800（一段文字的长度）

overlap 的作用：
防止关键信息被切在两块的边界。
比如"Python的内存管理机制采用引用计数..."如果刚好被切开，
"Python的内存管理机制"在前一块，后面细节在后一块，检索时可能丢上下文。
overlap=50 让相邻块有重叠，保证关键信息完整。
"""

import re
from typing import List, Dict
from io import BytesIO


class DocumentChunker:
    # ---- 解析各种格式 ----
    def parse(self, content: bytes, suffix: str) -> str:
        parsers = {
            "pdf": self._parse_pdf,
            "docx": self._parse_docx,
            "doc": self._parse_docx,
            "txt": self._parse_txt,
            "md": self._parse_txt,
        }
        parser = parsers.get(suffix, self._parse_txt)
        return parser(content)

    def _parse_pdf(self, content: bytes) -> str:
        """PDF 解析（懒加载 PyPDF2）"""
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            try:
                from pypdf import PdfReader
            except ImportError:
                raise ImportError(
                    "PDF 解析需要 PyPDF2 或 pypdf。安装: pip install PyPDF2"
                )
        reader = PdfReader(BytesIO(content))
        text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text.append(t)
        return "\n".join(text)

    def _parse_docx(self, content: bytes) -> str:
        """Word 文档解析（懒加载 python-docx）"""
        try:
            import docx
        except ImportError:
            raise ImportError(
                "Word 文档解析需要 python-docx。安装: pip install python-docx"
            )
        doc = docx.Document(BytesIO(content))
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    def _parse_txt(self, content: bytes) -> str:
        return content.decode("utf-8", errors="ignore")

    # ---- 分块策略 ----
    # 目录树和 ASCII 图使用的框线字符
    BOX_CHARS = set("├└│┌┐┘┤┬┴┼─━┃┏┓┗┛▼▲◀▶")

    def _clean(self, text: str) -> str:
        """行级清洗：剔除目录树、框线、分隔线。

        必须在切块之前做。只在块级过滤是不够的——目录树经常和正文
        被分到同一块里（中间没有空行），这时整块的字符占比仍然合格，
        块级过滤拦不住，`├── main.py` 就跟着正文一起入库了。

        判定用框线字符密度而不是整体有效字符比例：像 `├── app.py` 这种行
        含英文文件名，有效字符比例能到 0.45，按比例判不出来。但框线字符
        占了非空白字符的 30%，这个特征很稳定。
        """
        kept = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                kept.append("")
                continue
            # 纯分隔线：=== ----- *** 这类，字符种类不超过两种
            if len(s) >= 3 and len(set(s)) <= 2:
                continue
            nonspace = s.replace(" ", "")
            box = sum(1 for ch in s if ch in self.BOX_CHARS)
            # 框线字符达到 2 个且占比超过 20%，判为目录树/示意图
            if box >= 2 and nonspace and box / len(nonspace) > 0.2:
                continue
            # 兜底：有效字符比例过低（纯符号行）
            if self._content_ratio(s) < 0.35:
                continue
            kept.append(line)
        return "\n".join(kept)

    def split(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        分块策略（面试重点）：
        1. 行级清洗，剔掉目录树和分隔线
        2. 先按段落切分（保持语义单元完整）
        3. 每个段落如果还太长，按句子切
        4. 相邻块重叠 overlap 个字符

        进阶讨论（加分项）：
        - 还可以按标题层级切（Markdown的 # ## 结构）
        - 可以用语义分块（sentence-transformers 判断语义边界）
        """
        text = self._clean(text)
        # 按段落切分
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # 如果当前段落加上新段落不超限
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n"
            else:
                # 当前块保存
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # 如果段落本身太长，按句子切；无标点时按固定大小切
                if len(para) > chunk_size:
                    sentences = re.split(r'[。！？.!?]', para)
                    # 如果只有一个句子（即没有标点），按固定大小强制切割
                    if len(sentences) == 1:
                        for i in range(0, len(para), chunk_size):
                            chunks.append(para[i:i+chunk_size].strip())
                        current_chunk = ""
                    else:
                        sub_chunk = ""
                        for s in sentences:
                            chunk_candidate = s + "。"
                            if len(sub_chunk) + len(chunk_candidate) <= chunk_size:
                                sub_chunk += chunk_candidate
                            else:
                                if sub_chunk:
                                    chunks.append(sub_chunk.strip())
                                # 如果单个"句子"仍超过chunk_size，强制切割
                                if len(chunk_candidate) > chunk_size:
                                    for i in range(0, len(chunk_candidate), chunk_size):
                                        chunks.append(chunk_candidate[i:i+chunk_size].strip())
                                    sub_chunk = ""
                                else:
                                    sub_chunk = chunk_candidate
                        if sub_chunk:
                            current_chunk = sub_chunk
                else:
                    current_chunk = para + "\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        # 过滤低信息密度块
        # 真实文档里有大量目录树、ASCII 流程图、分隔线，切块之后会变成
        # 一堆 │ ─ ┌ ▼ 组成的碎片。它们有两个害处：一是占用 top_k 名额，
        # 二是向量几乎没有语义，却参与相似度计算，把分数分布拉偏。
        chunks = [c for c in chunks if self._is_meaningful(c)]

        # 添加重叠
        if overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_end = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
                overlapped.append(prev_end + "\n" + chunks[i])
            chunks = overlapped

        return chunks

    @staticmethod
    def _content_ratio(text: str) -> float:
        """有效字符（中文 / 字母 / 数字）占全文的比例。"""
        if not text:
            return 0.0
        meaningful = sum(
            1 for ch in text
            if ch.isalnum() or "\u4e00" <= ch <= "\u9fff"
        )
        return meaningful / len(text)

    @staticmethod
    def _is_meaningful(text: str, min_len: int = 40, min_ratio: float = 0.35) -> bool:
        """判断一个块是否值得入库。

        阈值取 0.35 而不是更高，是因为代码块和表格的有效字符比例天然偏低，
        过滤太激进会把真正的技术内容也扔掉。目录树的典型比例在 0.1 以下，
        0.35 足以把两类分开。
        """
        t = text.strip()
        return len(t) >= min_len and DocumentChunker._content_ratio(t) >= min_ratio
