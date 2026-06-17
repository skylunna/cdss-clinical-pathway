"""
嵌入式烟雾测试。
表明嵌入模型能够捕捉语义相似性：  
    - 关于同一主题的句子应具有较小的余弦距离  
    - 关于无关主题的句子应具有较大的余弦距离
使用方法：
    uv run python scripts/test_embedder.py
"""
import math

from cdss.core.logging import configure_logging, get_logger
from cdss.rag.embedder import get_embedder


logger = get_logger(__name__)


def cosine_distance(a: list[float], b: list[float]) -> float:
    """单位归一化向量的余弦距离（范围 [0, 2]）。"""
    # 对于归一化向量：余弦相似度 = = 点积
    sim = sum(x * y for x, y in zip(a, b, strict=True))
    # 用于处理浮点数的边缘情况
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


def main() -> None:
    configure_logging()


    sentences = [
        "社区获得性肺炎的经验性抗生素治疗方案",          # 0: CAP 治疗
        "CAP 患者首选阿莫西林口服",                       # 1: CAP 用药 (与0相关)
        "青霉素过敏患者应避免使用 β-内酰胺类药物",       # 2: 过敏
        "对青霉素有严重过敏反应的病人不能用 β-内酰胺",   # 3: 过敏 (与2近义)
        "CURB-65 评分用于评估肺炎严重度",                # 4: 评分
        "深度学习模型在图像分类中的应用",                # 5: 完全无关
    ]

    # --- Embed all ---
    embedder = get_embedder()
    logger.info("embedding_sentences", count=len(sentences))
    vectors = embedder.embed_batch(sentences)
    logger.info("embedded", dim=len(vectors[0]))

    # ----- Distance matrix -----
    print("\n" + "=" * 80)
    print("Cosine distance matrix (smaller = more similar)")
    print("=" * 80)

        # Header
    print(f"\n{'':>4}", end="")
    for j in range(len(sentences)):
        print(f"  [{j}]  ", end="")
    print()

    for i, vi in enumerate(vectors):
        print(f"[{i}] ", end="")
        for vj in vectors:
            d = cosine_distance(vi, vj)
            print(f" {d:.3f} ", end="")
        print(f"  {sentences[i][:30]}")

    # ----- Highlights -----
    print("\n" + "=" * 80)
    print("Highlights")
    print("=" * 80)

    pairs_to_check = [
        (0, 1, "Both about CAP treatment"),
        (2, 3, "Both about penicillin allergy"),
        (0, 5, "CAP vs unrelated (deep learning)"),
        (2, 5, "Allergy vs unrelated"),
        (0, 4, "CAP treatment vs CURB-65 (related but different aspects)"),
    ]
    for i, j, label in pairs_to_check:
        d = cosine_distance(vectors[i], vectors[j])
        print(f"  dist([{i}], [{j}]) = {d:.4f}   ← {label}")


if __name__ == "__main__":
    main()