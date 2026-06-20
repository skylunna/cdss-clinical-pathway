"""
Markdown-aware document chunker.

Strategy:
1. Parse markdown headings; each section becomes a candidate chunk
2. Sections short enough -> kept as-is (preserves structural boundaries)
3. Sections too long -> further split with overlapping windows,
   preferring sentence/paragraph boundaries

The output `ChunkData` carries the section path (e.g., "经验性抗生素治疗 > ICU 治疗")
so retrieval results can be presented with context, and downstream consumers
(citations, audit) can trace each chunk back to its position in the source.

支持 Markdown 的文档分块器。
策略：
    1. 解析 Markdown Markdown 标题；每个部分都成为候选片段
    2. 节段较短 -> 保持原样（保留结构边界）
    3. 段落过长时，使用重叠窗口进一步分割，优先保留句子或段落的边界。
输出 `ChunkData`` 包含章节路径（例如：“经验性抗生素治疗 > ICU ICU ICU 治疗”），
以便检索结果能够提供上下文信息，下游用户（如引用、审计）可将每个片段追溯到其在源文本中的位置。
"""
from dataclasses import dataclass


@dataclass
class ChunkData:
    """在嵌入和持久化之前生成的块。"""

    content: str
    source: str
    section: str | None
    chunk_index: int


class MarkdownChunker:
    """按结构划分，再按大小分割 Markdown 文本。"""

    def __init__(self, target_size: int = 400, overlap: int = 50) -> None:
        # 安全校验：重叠长度不能大于等于单块长度，否则逻辑实效，抛出异常
        if overlap >= target_size:
            raise ValueError("overlap must be smaller than target_size")
        self.target_size = target_size  # 单块文本最大字符长度 (默认400)
        self.overlap = overlap  # 相邻两块的重叠字符数 (默认 50)，解决分段丢失上下文的问题

    def chunk(self, text: str, source: str) -> list[ChunkData]:
        """返回给定文本的分块列表。"""
        # 把整篇文档按章节 / 标题 切分成多个小节，返回格式 (section_path, content)
        # section_path: 章节路径，比如 第一章/肺炎诊断标准，存入数据库 section 字段
        # content: 该章节的纯文本内容
        sections = self._split_into_sections(text)

        chunks: list[ChunkData] = []
        index = 0
        for section_path, content in sections:
            if not content:
                continue    # 空章节直接跳过
            if len(content) <= self.target_size:
                # 400 字符内
                # 整段直接作为一个知识库分片，生成一条 knowledge_chunks 记录
                chunks.append(
                    ChunkData(
                        content=content,
                        source=source,  # 文档来源文件
                        section=section_path,   # 所属章节
                        chunk_index=index,  # 当前全局分片序号
                    )
                )
                index += 1
            else:
                for sub in self._split_long(content):
                    # 文本超长，超过 400 字符，调用 _split_long 重叠分片
                    chunks.append(
                        ChunkData(
                            content=sub,
                            source=source,
                            section=section_path,
                            chunk_index=index,
                        )
                    )
                    index += 1

        return chunks
    
    # ---------- internals ----------
    def _split_into_sections(self, text: str) -> list[tuple[str | None, str]]:
        """Walk lines, track heading stack, emit (section_path, content) for each section."""
        # 走线、追踪方向堆栈，为每个部分分别发射（section_path, content）
        # 按 Markdown 标题 (# 一级、## 二级...) 拆分文档为层级章节，生成章节路径 + 章节正文
        # _parse_heading: 辅助识别一行文本是不是Markdown标题，返回标题层级与标题文字
        # _split_long: 超长章节文本做滑动窗口重叠切分，优先在句号/换行等自然语义断点切割，不生硬截断句子
        section_stack: list[str] = []
        buffer: list[str] = []
        sections: list[tuple[str | None, str]] = []

        def flush() -> None:
            if buffer:
                path = " > ".join(section_stack) if section_stack else None
                content = "\n".join(buffer).strip()
                if content:
                    sections.append((path, content))

        for line in text.splitlines():
            level, heading_text = self._parse_heading(line)
            if level is not None:
                flush()
                buffer.clear()
                # Pop deeper headings off the stack, then push the new one
                # 从栈中弹出更深层的标题，然后压入新的标题
                while len(section_stack) >= level:
                    section_stack.pop()
                section_stack.append(heading_text)  # type: ignore[arg-type]
            else:
                buffer.append(line)
        
        flush()
        return sections
    
    @staticmethod
    def _parse_heading(line: str) -> tuple[int | None, str | None]:
        """Returns (level, text) if line is a markdown heading, else (None, None)."""
        # 如果行是 Markdown Markdown 标题，则返回 (level, text)，否则返回 (None, None)。
        stripped = line.strip()
        if not stripped.startswith("#"):
            return None, None
        level = 0
        while level < len(stripped) and stripped[level] == "#":
            level += 1
        if 1 <= level <= 6 and len(stripped) > level and stripped[level] == " ":
            return level, stripped[level + 1 :].strip()
        return None, None
    
    def _split_long(self, text: str) -> list[str]:
        """Split text into overlapping windows, preferring natural boundaries."""
        # 将文本分割成重叠的窗口，优先选择自然分界线。
        if len(text) <= self.target_size:
            return [text]
        
        chunks: list[str] = []
        start = 0
        while start < len(text):
            # 暂定窗口末尾
            end = min(start + self.target_size, len(text))
            if end < len(text):
                # 从end往回找天然分隔符，优先级：段落换行 >> 句号换行 >> 句号 >> 普通换行
                for boundary in ["\n\n", "。\n", "。", "\n", "; ", ";"]:
                    idx = text.rfind(boundary, start, end)
                    # 分割点不能太靠近窗口开头，(过半才采用，避免块太短)
                    if idx > start + self.target_size // 2:
                        end = idx + len(boundary)
                        break
            # 截取片段、去空白
            sub = text[start:end].strip()
            if sub:
                chunks.append(sub)
            if end >= len(text):
                break
            # 下一段起点 = 当前结尾 - 重叠长度，实现上下文重叠
            start = max(end - self.overlap, start + 1)

        return chunks