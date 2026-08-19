import os
import os.path as osp
import json
import math
import re
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), ".env"))

client_userInfo = OpenAI(
    api_key=os.getenv("USERINFO_API_KEY"),
    base_url=os.getenv("USERINFO_BASE_URL"),
)
USERINFO_MODEL_NAME = os.getenv("USERINFO_MODEL_NAME")



BASE_DIR = osp.dirname(osp.abspath(__file__))
# ---------------- 配置（重点：指向你的原始QA JSON文件） ----------------
LOCAL_MODEL_PATH = osp.join(BASE_DIR, "./models/m3e")  # 你的模型路径
MILVUS_DB_PATH = osp.join(BASE_DIR, "sichuan_m3e.db") # 与插入时一致的数据库路径
COLLECTION_NAME = "question"  # 与插入时一致的Collection名
DEFAULT_TOPN = 15  # 返回Top N个最相似结果（越多越慢，按需调整）
KEYWORD_TOPN = 30  # 关键词召回候选数，稍大一些，交给后续LLM精排
HYBRID_RRF_K = 60  # RRF常用平滑参数，降低单一路径排名波动
VECTOR_RRF_WEIGHT = 1.0
KEYWORD_RRF_WEIGHT = 1.2  # 高考场景学校名/专业名很关键，关键词结果略微加权
QUERY_PREFIX = "为问题生成嵌入："  # 必须与插入时的前缀一致（关键！）
FILTER_EXPR = "source == 'rag_questions.json'"  # 精准过滤JSON来源数据（提速核心）

# 核心配置：你的原始QA JSON文件路径（直接从中提取q->a映射）
RAW_QA_JSON_FILE = osp.join(BASE_DIR, "final_knowledge.json") 
# ----------------------------------------------------------------------

# 全局缓存：加载一次QA映射，后续查询直接复用（避免重复读文件，大幅提速）
QA_MAPPING_CACHE = None
KEYWORD_INDEX_CACHE = None


def init_model(local_model_path):
    """初始化模型（仅加载一次，避免重复开销）"""
    if not osp.exists(local_model_path) or not osp.isdir(local_model_path):
        raise FileNotFoundError(f"模型路径不存在/非目录：{local_model_path}")
    return SentenceTransformer(
        model_name_or_path=local_model_path,
        trust_remote_code=True
    )


def init_milvus_client(db_path):
    """初始化Milvus客户端（仅加载一次）"""
    return MilvusClient(db_path)


def generate_embeddings(model, text_list, is_query=False, query_prefix=""):
    """生成向量（与插入时逻辑完全一致，确保匹配度）"""
    if is_query:
        texts = [f"{query_prefix}{text}" for text in text_list]
    else:
        texts = text_list
    return model.encode(
        sentences=texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )


def load_raw_qa_mapping(json_path):
    """
    直接从原始QA JSON提取question->answer映射（核心修改）
    一次加载后缓存，后续查询无需重复解析文件
    """
    global QA_MAPPING_CACHE
    if QA_MAPPING_CACHE is not None:
        print(f"✅ 复用缓存的QA映射（共{len(QA_MAPPING_CACHE)}条）")
        return QA_MAPPING_CACHE

    # 首次加载：解析原始JSON
    if not osp.exists(json_path):
        raise FileNotFoundError(f"原始QA JSON文件不存在：{json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_qa_list = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON格式错误：{e}（请检查文件语法）")

    # 验证数据结构（确保是列表，且每个元素含question和answer）
    if not isinstance(raw_qa_list, list):
        raise TypeError(f"JSON根节点必须是列表，当前是{type(raw_qa_list).__name__}")

    qa_mapping = {}
    for idx, item in enumerate(raw_qa_list, 1):
        if not isinstance(item, dict) or "question" not in item or "answer" not in item:
            print(f"⚠️  跳过无效数据（第{idx}条）：缺少question/answer字段")
            continue

        # 处理问题字段
        question = item["question"]
        if isinstance(question, str):
            question = question.strip()
        elif isinstance(question, dict):
            # 如果question是字典，尝试提取文本或转换为字符串
            question = str(question).strip()
            # print(f"⚠️  问题字段是字典，已转换为字符串：{question[:50]}...")
        else:
            question = str(question).strip()

        # 处理答案字段
        answer = item["answer"]
        if isinstance(answer, str):
            answer = answer.strip()
        elif isinstance(answer, dict):
            # 如果answer是字典，尝试提取文本内容或转换为JSON字符串
            # 优先查找常见的文本字段
            text_fields = ["text", "content", "answer", "response", "答案", "内容"]
            for field in text_fields:
                if field in answer and isinstance(answer[field], str):
                    answer = answer[field].strip()
                    break
            else:
                # 如果没有找到文本字段，将整个字典转换为JSON字符串
                answer = json.dumps(answer, ensure_ascii=False, indent=2)
            # print(f"⚠️  答案字段是字典，已处理：{answer[:50]}...")
        elif isinstance(answer, list):
            # 如果answer是列表，转换为字符串
            answer = " ".join(str(x) for x in answer)
        else:
            answer = str(answer)

        # 去重：若有重复问题，保留最后一条的答案
        if question:  # 过滤空问题
            qa_mapping[question] = answer if answer else "暂无数据"

    if not qa_mapping:
        raise ValueError("原始JSON中未提取到有效QA对")

    # 缓存映射，后续查询直接用
    QA_MAPPING_CACHE = qa_mapping
    print(f"✅ 首次加载QA映射完成（共{len(qa_mapping)}条有效数据）")
    return qa_mapping


def normalize_for_keyword(text):
    """关键词检索用的轻量归一化：统一大小写并去掉空白。"""
    return re.sub(r"\s+", "", str(text or "").lower())


def tokenize_for_keyword(text):
    """
    中文关键词召回分词。
    优先使用jieba；环境没有jieba时，用中英文数字抽取 + 中文2/3/4-gram兜底。
    这样学校名、专业名、选科、批次等实体词不会完全依赖向量语义。
    """
    text = normalize_for_keyword(text)
    if not text:
        return []

    tokens = []
    try:
        import jieba
        tokens.extend(t for t in jieba.lcut(text) if len(t.strip()) >= 2)
    except ImportError:
        pass

    for part in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text):
        if re.fullmatch(r"[a-z0-9]+", part):
            tokens.append(part)
            continue

        if len(part) >= 2:
            if len(part) <= 8:
                tokens.append(part)
            for n in (2, 3, 4):
                if len(part) >= n:
                    tokens.extend(part[i:i + n] for i in range(len(part) - n + 1))

    return tokens


def build_keyword_index(qa_mapping):
    """基于问题文本构建一个内存BM25索引。"""
    documents = []
    inverted_index = defaultdict(list)
    doc_lengths = []

    for question in qa_mapping.keys():
        token_counts = Counter(tokenize_for_keyword(question))
        if not token_counts:
            continue

        doc_id = len(documents)
        documents.append(question)
        doc_lengths.append(sum(token_counts.values()))
        for token, freq in token_counts.items():
            inverted_index[token].append((doc_id, freq))

    avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
    return {
        "documents": documents,
        "inverted_index": dict(inverted_index),
        "doc_lengths": doc_lengths,
        "avg_doc_len": avg_doc_len,
        "doc_count": len(documents),
    }


def get_keyword_index():
    """延迟加载关键词索引，避免服务启动时就解析大JSON。"""
    global KEYWORD_INDEX_CACHE
    if KEYWORD_INDEX_CACHE is None:
        qa_mapping = load_raw_qa_mapping(RAW_QA_JSON_FILE)
        KEYWORD_INDEX_CACHE = build_keyword_index(qa_mapping)
        print(f"✅ 关键词索引构建完成（共{KEYWORD_INDEX_CACHE['doc_count']}个问题）")
    return KEYWORD_INDEX_CACHE


def keyword_search(query_text, topn=KEYWORD_TOPN):
    """
    本地关键词召回。返回结构对齐Milvus结果，方便后续混合融合。
    """
    keyword_index = get_keyword_index()
    doc_count = keyword_index["doc_count"]
    if doc_count == 0:
        return []

    query_tokens = Counter(tokenize_for_keyword(query_text))
    if not query_tokens:
        return []

    inverted_index = keyword_index["inverted_index"]
    doc_lengths = keyword_index["doc_lengths"]
    avg_doc_len = keyword_index["avg_doc_len"] or 1
    k1 = 1.5
    b = 0.75
    scores = defaultdict(float)

    for token, query_freq in query_tokens.items():
        postings = inverted_index.get(token)
        if not postings:
            continue

        df = len(postings)
        idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
        for doc_id, term_freq in postings:
            doc_len = doc_lengths[doc_id] or 1
            denom = term_freq + k1 * (1 - b + b * doc_len / avg_doc_len)
            scores[doc_id] += query_freq * idf * (term_freq * (k1 + 1) / denom)

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:topn]
    documents = keyword_index["documents"]
    return [
        {
            "entity": {"text": documents[doc_id], "source": "keyword_bm25"},
            "distance": score,
            "keyword_score": score,
        }
        for doc_id, score in ranked
    ]


def extract_question_from_result(result):
    entity = result.get("entity", {}) if isinstance(result, dict) else {}
    return entity.get("text", "").strip()


def fuse_search_results(vector_results=None, keyword_results=None, limit=DEFAULT_TOPN):
    """
    使用Reciprocal Rank Fusion融合向量召回和关键词召回。
    RRF只依赖排序名次，不要求两路分数同尺度，适合Milvus相似度和BM25分数混合。
    """
    fused = {}

    def add_results(results, weight, channel):
        for rank, result in enumerate(results or [], 1):
            question = extract_question_from_result(result)
            if not question:
                continue
            if question not in fused:
                fused[question] = {"score": 0.0, "result": result, "channels": []}
            fused[question]["score"] += weight / (HYBRID_RRF_K + rank)
            fused[question]["channels"].append(channel)

    add_results(vector_results, VECTOR_RRF_WEIGHT, "vector")
    add_results(keyword_results, KEYWORD_RRF_WEIGHT, "keyword")

    ranked_items = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
    final_results = []
    for item in ranked_items[:limit]:
        result = item["result"]
        result["hybrid_score"] = item["score"]
        result["hybrid_channels"] = item["channels"]
        final_results.append(result)
    return final_results


def filter_with_llm(query_text, candidate_questions):
    """
    使用大模型API筛选与查询最相关的候选问题
    返回：相关问题的编号列表（从1开始）
    """
    if not candidate_questions:
        return []
    
    # 格式化候选问题：给每个问题加上编号
    numbered_questions = []
    for idx, question in enumerate(candidate_questions, 1):
        numbered_questions.append(f"{idx}. {question}")
    
    formatted_candidates = "\n".join(numbered_questions)
    
    print("原RAG查询结果（已编号）：")
    print(formatted_candidates)
    

    # 构建大模型输入（优化版提示词）
    prompt = f"""你是一个专业的问题匹配助手。请分析用户查询与以下候选问题的语义相关性，筛选出真正相关的问题编号。

**用户查询：**
{query_text}

**候选问题列表：**
{formatted_candidates}

**任务要求：**
1. 仔细分析用户查询的核心意图和关键信息
2. 筛选出与用户查询语义相关、能够回答用户问题的候选问题
3. 只输出相关问题的编号，用逗号分隔（例如：1,3,5）
4. 如果所有候选问题都不相关，请输出"无"
5. 不要输出任何解释或其他内容，只输出编号或"无"
6. “物理类”是指“高考选科为物理”，历史类同理，“物理学”指物理学专业。

**输出格式示例：**
- 有相关问题：1,3,5
- 无相关问题：无

请输出："""

    try:
        response = client_userInfo.chat.completions.create(
            model=USERINFO_MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个专业的问题匹配助手，擅长分析语义相关性。请严格按照用户要求输出结果。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  # 降低温度，使输出更稳定
        )

        response_text = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"⚠️ 大模型调用失败: {str(e)}")
        # 失败时返回空列表，避免程序中断
        return []
    
    print("✅ 大模型输出:", response_text)
    

    # 解析大模型返回的编号（更健壮的解析逻辑）
    # 1. 先检查是否明确表示"无相关"
    no_match_keywords = ["无", "none", "没有", "无相关", "不相关"]
    if any(keyword in response_text.lower() for keyword in no_match_keywords):
        print("🤖 大模型判断：无相关问题")
        return []

    # 2. 提取所有数字
    import re
    numbers = re.findall(r'\d+', response_text)
    
    if not numbers:
        print("⚠️ 大模型输出中未找到有效编号，返回空结果")
        return []
    
    # 3. 过滤有效的编号（必须在 1 到候选问题数量范围内）
    selected_indices = []
    for num_str in numbers:
        try:
            num = int(num_str)
            if 1 <= num <= len(candidate_questions):
                selected_indices.append(num)
            else:
                print(f"⚠️ 忽略超出范围的编号: {num}")
        except ValueError:
            continue
    
    # 4. 去重并排序
    selected_indices = sorted(list(set(selected_indices)))
    
    if not selected_indices:
        print("⚠️ 未找到有效的问题编号")
        return []

    return selected_indices

def fast_rag_search(client, collection, model, query_text, topn=DEFAULT_TOPN, filter_expr=None):
    """
    快速查询核心函数
    """
    vector_results = []

    # 1. Milvus向量召回：可用时作为语义召回，不可用时退化为关键词召回。
    if client is not None and model is not None:
        try:
            query_embedding = generate_embeddings(
                model, text_list=[query_text], is_query=True, query_prefix=QUERY_PREFIX
            )
            search_params = {
                "collection_name": collection,
                "data": query_embedding.tolist(),
                "limit": topn,
                "output_fields": ["text", "source"],
                "consistency_level": "Strong"
            }

            if filter_expr:
                search_params["filter"] = filter_expr

            search_res = client.search(**search_params)
            vector_results = search_res[0] if isinstance(search_res, list) else search_res
            print(f"🔍 Milvus检索到{len(vector_results)}个候选问题")
        except Exception as e:
            print(f"⚠️ Milvus向量检索失败，改用关键词召回：{str(e)}")

    # 2. 本地关键词召回：补足学校名、专业名、选科等精确实体命中。
    try:
        keyword_results = keyword_search(query_text, topn=max(KEYWORD_TOPN, topn))
        print(f"🔎 关键词检索到{len(keyword_results)}个候选问题")
    except Exception as e:
        keyword_results = []
        print(f"⚠️ 关键词检索失败：{str(e)}")

    # 3. 融合两路结果，再交给原有LLM筛选。
    results = fuse_search_results(vector_results, keyword_results, limit=max(topn, DEFAULT_TOPN))

    if not results:
        return "❌ 未找到相似问题"

    # 提取候选问题文本
    candidate_questions = [extract_question_from_result(res) for res in results]
    candidate_questions = [question for question in candidate_questions if question]
    print(f"✅ 混合召回融合后得到{len(candidate_questions)}个候选问题")

    # 4. 使用大模型API筛选相关候选问题
    selected_indices = filter_with_llm(query_text, candidate_questions)

    if not selected_indices:
        return "本地知识库未命中相关问题，请尝试联网查询"

    print(f"✅ 大模型筛选出{len(selected_indices)}个相关问题：{selected_indices}")

    # 5. 根据筛选结果匹配答案
    qa_mapping = load_raw_qa_mapping(RAW_QA_JSON_FILE)

    result_str = f"找到{len(selected_indices)}个相关回答：\n\n"

    for idx, res_idx in enumerate(selected_indices, 1):
        if 1 <= res_idx <= len(results):
            res = results[res_idx - 1]  # 转换为0-based索引
            entity = res.get("entity", {})
            matched_question = entity.get("text", "<未找到问题>").strip()
            matched_answer = qa_mapping.get(matched_question, "⚠️ 未匹配到答案")

            result_str += f"--- 回答{idx}（原优先级{res_idx}）---\n"
            result_str += f"问题：{matched_question}\n"
            result_str += f"答案：{matched_answer}\n\n"
        else:
            print(f"⚠️  无效的索引：{res_idx}")

    return result_str


# ---------------- 对外查询接口（一次初始化，多次复用） ----------------
def knowledge_search(query_text: str = ""):
    """
    对外调用接口：
    - 模型/客户端只初始化一次（避免重复加载，大幅提速）
    - 自动处理缓存和映射
    """
    if not query_text.strip():
        return "❌ 请输入有效的查询问题"

    # 全局变量：确保模型和客户端只加载一次（核心提速点）
    global MODEL, MILVUS_CLIENT
    try:
        # 首次调用：初始化模型和客户端
        MODEL
        MILVUS_CLIENT
    except NameError:
        print("🔧 首次初始化模型和Milvus客户端...")
        MODEL = None
        MILVUS_CLIENT = None
        try:
            MODEL = init_model(LOCAL_MODEL_PATH)
            print("模型初始化完成")
            MILVUS_CLIENT = init_milvus_client(MILVUS_DB_PATH)
            print("milvus连接完成")
            # 验证Collection是否存在
            if not MILVUS_CLIENT.has_collection(COLLECTION_NAME):
                print(f"⚠️ Collection不存在：{COLLECTION_NAME}，本次仅使用关键词召回")
                MODEL = None
                MILVUS_CLIENT = None
        except Exception as e:
            print(f"⚠️ 向量检索初始化失败，本次仅使用关键词召回：{str(e)}")

    # 执行快速查询
    return fast_rag_search(MILVUS_CLIENT, COLLECTION_NAME, MODEL, query_text)


# ---------------- 测试 ----------------
if __name__ == "__main__":
    #测试1：首次查询（会初始化模型、加载映射）
    print("=== 测试1：首次查询 ===")
    res1 = knowledge_search("电子科技大学有哪些专业？")
    print(res1)

     # 测试2：重复查询（复用缓存，速度更快）
    print("=== 测试2：重复查询（复用缓存） ===")
    res2 = knowledge_search("电子科技大学计算机专业的录取分数是多少？")
    print(res2)
