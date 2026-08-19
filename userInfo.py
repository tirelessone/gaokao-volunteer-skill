import time
import json
from typing import Dict, Optional, Union
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

# 初始化OpenAI客户端
client_userInfo = OpenAI(
    api_key=os.getenv("USERINFO_API_KEY"),
    base_url=os.getenv("USERINFO_BASE_URL")
)
USERINFO_MODEL_NAME = os.getenv("USERINFO_MODEL_NAME")

# -------------------------------------------用户输入----------------------------------------------------#
"""示例信息：如果用户提供了信息，则使用用户提供的信息"""
# major="电子信息类" #意向专业
# score=630 #总分
# chinese=100 #语文
# math=100 #数学
# foreign_language=["英", 100] #外语
# reselection_1=["物", 100] #副科1
# reselection_2=["化", 100] #副科2
# reselection_3=["生", 100] #副科3
# foreign=0 # 中外合作，1=考虑 0=不考虑
# city = ["北京","上海","成都","广东","浙江"] # 意向城市，省市都行
# university=["电子科技大学"] #意向学校
# is_color_blindness=0 #色盲，0=否 1=是

"""默认信息：如果用户没有提供信息，则使用默认信息"""
# major=null #意向专业
# score=null #总分
# chinese=null #语文
# math=null #数学
# foreign_language=null #外语
# reselection_1=null #副科1
# reselection_2=null #副科2
# reselection_3=null #副科3
# foreign=null # 中外合作，1=考虑 0=不考虑
# city = null # 意向城市，省市都行
# university=null #意向学校
# is_color_blindness=null #色盲，0=否 1=是

# -------------------------------------------请求api需要----------------------------------------------------#
firstChoice = "物"  # "物"或"史"
recruit = 0  # 0=本科批 1=专科批
province = "四川省"  # 学生高考省份
reselection = ["不限", "化", "生"]  # 请求api需要
rate_min = 5  # 请求api时需要的概率区间
rate_max = 90


def extract_city_name(user_description: str) -> Optional[list]:
    """
    根据用户描述提取具体的城市或省份全称列表，支持模糊说法映射

    Args:
        user_description (str): 用户对城市的描述（支持"沿海城市"、"新一线城市"等模糊说法）

    Returns:
        Optional[list]: 提取的城市/省份全称列表，失败时返回None
    """
    if not user_description or user_description.strip() == "":
        return None

    city_mapping_prompt = f"""你是一名地理信息助手。你的任务是根据用户的描述，将模糊的城市/地区说法映射为具体的省份或城市全称列表。

⚠️ 要求：
1. 输出必须使用完整的省份或城市名称，如"北京市"、"上海市"、"四川省"、"江苏省"等。
2. 输出必须是 JSON 数组格式，例如：["北京市", "上海市"] 或 ["四川省"]。
3. 不要输出任何解释、额外文字或符号。
4. 如果无法判断或无法映射，应输出 ["无法匹配"]。
5. 对于模糊说法，请根据常见理解进行映射：
   - "沿海地区/沿海城市"：包括所有沿海省份和直辖市
   - "一线城市"：北京市、上海市、广州市、深圳市
   - "新一线城市"：成都市、杭州市、重庆市、西安市、苏州市、武汉市、南京市、天津市、郑州市、长沙市、东莞市、沈阳市、青岛市、合肥市、佛山市
   - "二线城市"：昆明市、大连市、厦门市、哈尔滨市、济南市、福州市、南宁市、温州市、石家庄市、长春市、泉州市、贵阳市、南昌市、金华市、常州市、珠海市、惠州市、嘉兴市、南通市、中山市、保定市、兰州市、台州市、徐州市、太原市、绍兴市、烟台市、廊坊市
   - "川渝地区"：四川省、重庆市
   - "京津冀"：北京市、天津市、河北省
   - "长三角"：上海市、江苏省、浙江省、安徽省
   - "珠三角"：广东省（特别是广州市、深圳市、珠海市、佛山市、东莞市、中山市等）
   - "东北地区"：辽宁省、吉林省、黑龙江省
   - "西北地区"：陕西省、甘肃省、青海省、宁夏回族自治区、新疆维吾尔自治区
   - "西南地区"：四川省、重庆市、贵州省、云南省、西藏自治区
   - "华北地区"：北京市、天津市、河北省、山西省、内蒙古自治区
   - "华东地区"：上海市、江苏省、浙江省、安徽省、福建省、江西省、山东省
   - "华南地区"：广东省、广西壮族自治区、海南省
   - "华中地区"：河南省、湖北省、湖南省
   - "偏远城市": 黑河市,伊春市,乌兰察布市,固原市,定西市,玉树藏族自治州,喀什地区,阿里地区,怒江傈僳族自治州,黔西南布依族苗族自治州,河池市,巴中市
6. 如果用户明确列出具体城市，直接补充完整的省市名称（如"北京"→"北京市"，"成都"→"成都市"，"广东"→"广东省"）

【中国省级行政区完整列表】（供参考）
直辖市：北京市、上海市、天津市、重庆市
省份：河北省、山西省、辽宁省、吉林省、黑龙江省、江苏省、浙江省、安徽省、福建省、江西省、山东省、河南省、湖北省、湖南省、广东省、海南省、四川省、贵州省、云南省、陕西省、甘肃省、青海省、台湾省
自治区：内蒙古自治区、广西壮族自治区、西藏自治区、宁夏回族自治区、新疆维吾尔自治区
特别行政区：香港特别行政区、澳门特别行政区

【示例】
用户：沿海地区
输出：["辽宁省", "河北省", "天津市", "山东省", "江苏省", "上海市", "浙江省", "福建省", "广东省", "广西壮族自治区", "海南省"]

用户：新一线城市
输出：["成都市", "杭州市", "重庆市", "西安市", "苏州市", "武汉市", "南京市", "天津市", "郑州市", "长沙市", "东莞市", "沈阳市", "青岛市", "合肥市", "佛山市"]

用户：除了北京、上海的一线城市
输出：["广州市", "深圳市"]

用户：川渝地区
输出：["四川省", "重庆市"]

用户：北京、成都、广东
输出：["北京市", "成都市", "广东省"]

用户：长三角
输出：["上海市", "江苏省", "浙江省", "安徽省"]

用户：偏远城市
输出：["黑河市", "伊春市", "乌兰察布市", "固原市", "定西市", "玉树藏族自治州", "喀什地区", "阿里地区", "怒江傈僳族自治州", "黔西南布依族苗族自治州", "河池市", "巴中市"]

【用户输入】
{user_description}
"""

    try:
        response = client_userInfo.chat.completions.create(
            model=USERINFO_MODEL_NAME,
            messages=[
                {"role": "user", "content": city_mapping_prompt}
            ],
            temperature=0.7
        )

        response_text = response.choices[0].message.content.strip()

        # 尝试解析JSON数组
        try:
            # 查找JSON数组
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            if start != -1 and end > 0:
                json_str = response_text[start:end]
                city_list = json.loads(json_str)

                # 如果返回["无法匹配"]，则返回None
                if city_list == ["无法匹配"]:
                    print(f"⚠️ [DEBUG] 城市映射无法匹配: {user_description}")
                    return None

                return city_list
            else:
                # 如果没有找到JSON数组，尝试直接解析
                city_list = json.loads(response_text)
                if city_list == ["无法匹配"]:
                    print(f"⚠️ [DEBUG] 城市映射无法匹配: {user_description}")
                    return None
                return city_list

        except json.JSONDecodeError:
            # 如果JSON解析失败，返回None
            print(f"❌ [DEBUG] 城市映射JSON解析失败: {response_text}")
            return None

    except Exception as e:
        print(f"❌ [DEBUG] 城市映射提取失败: {str(e)}")
        return None


def extract_major_category(user_description: str) -> Optional[list]:
    """
    根据用户描述提取符合官方规定的专业大类列表

    Args:
        user_description (str): 用户对专业的描述

    Returns:
        Optional[list]: 提取的专业大类名称列表，失败时返回None
    """
    if not user_description or user_description.strip() == "":
        return None

    major_category_prompt = f"""你是一名专业分类助手。你的任务是根据用户的描述，推断出用户最可能的专业意向，并从以下【官方规定的大类列表】中，严格选出一个或多个最符合的专业大类。  

⚠️ 要求：  
1. 输出必须严格使用官方大类名称，不允许出现任何变形、同义词或自创类别。  
2. 输出必须是 JSON 数组，例如：["计算机类"] 或 ["临床医学类", "口腔医学类", "中西医结合类", "医学技术类"]。  
3. 不要输出任何解释、额外文字或符号。  
4. 如果无法判断，应输出 ["无法匹配"]。  

【官方规定的大类列表】  
1: 哲学类  
2: 经济学类  
3: 财政学类  
4: 金融学类  
5: 经济与贸易类  
6: 法学类  
7: 政治学类  
8: 社会学类  
9: 民族学类  
10: 马克思主义理论类  
11: 公安学类  
12: 教育学类  
13: 体育学类  
14: 中文语言文学类  
15: 外国语言文学类  
16: 新闻传播学类  
17: 历史学类  
18: 数学类  
19: 物理学类  
20: 化学类  
21: 天文学类  
22: 地理科学类  
23: 大气科学类  
24: 海洋科学类  
25: 地球物理学类  
26: 地质学类  
27: 生物科学类  
28: 心理学类  
29: 统计学类  
30: 力学类  
31: 机械类  
32: 仪器类  
33: 材料类  
34: 能源动力类  
35: 电气类  
36: 电子信息类  
37: 自动化类  
38: 计算机类  
39: 土木类  
40: 水利类  
41: 测绘类  
42: 化工与制药类  
43: 地质类  
44: 矿业类  
45: 纺织类  
46: 轻工类  
47: 交通运输类  
48: 海洋工程类  
49: 航空航天类  
50: 兵器类  
51: 核工程类  
52: 农业工程类  
53: 林业工程类  
54: 环境科学与工程类  
55: 生物医学工程类  
56: 食品科学与工程类  
57: 建筑类  
58: 安全科学与工程类  
59: 生物工程类  
60: 公安技术类  
61: 交叉工程类  
62: 植物生产类  
63: 自然保护与环境生态类  
64: 动物生产类  
65: 动物医学类  
66: 林学类  
67: 水产类  
68: 草学类  
69: 基础医学类  
70: 临床医学类  
71: 口腔医学类  
72: 公共卫生与预防医学类  
73: 中医学类  
74: 中西医结合类  
75: 药学类  
76: 中药学类  
77: 法医学类  
78: 医学技术类  
79: 护理学类  
80: 管理科学与工程类  
81: 工商管理类  
82: 农业经济管理类  
83: 公共管理类  
84: 图书情报与档案管理类  
85: 物流管理与工程类  
86: 工业工程类  
87: 电子商务类  
88: 旅游管理类  
89: 艺术学理论类  
90: 音乐与舞蹈学类  
91: 戏剧与影视学类  
92: 美术学类  
93: 设计学类  

【示例】  
用户：我想学计算机  
输出：["计算机类"]  

用户：我想学医  
输出：["临床医学类", "口腔医学类", "中西医结合类", "医学技术类"]  

用户：我想学机器人相关的专业
输出：["交叉工程类", "自动化类", "机械类"]

【用户输入】  
{user_description} 
"""

    try:
        response = client_userInfo.chat.completions.create(
            model=USERINFO_MODEL_NAME,
            messages=[
                {"role": "user", "content": major_category_prompt}
            ],
            temperature=0.7
        )

        response_text = response.choices[0].message.content.strip()

        # 尝试解析JSON数组
        try:
            # 查找JSON数组
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            if start != -1 and end > 0:
                json_str = response_text[start:end]
                major_categories = json.loads(json_str)

                # 如果返回["无法匹配"]，则返回None
                if major_categories == ["无法匹配"]:
                    return None

                return major_categories
            else:
                # 如果没有找到JSON数组，尝试直接解析
                major_categories = json.loads(response_text)
                if major_categories == ["无法匹配"]:
                    return None
                return major_categories

        except json.JSONDecodeError:
            # 如果JSON解析失败，返回None
            print(f"专业大类JSON解析失败: {response_text}")
            return None

    except Exception as e:
        print(f"专业大类提取失败: {str(e)}")
        return None


def format_user_info_with_llm(query: Union[str, Dict]) -> Optional[Dict]:
    """
    使用大模型提取和格式化用户输入的信息，或直接处理已格式化的用户信息

    Args:
        query (Union[str, Dict]): 用户输入的原始文本 或 已格式化的用户信息字典

    Returns:
        Optional[Dict]: 提取的用户信息字典，提取失败时返回None
    """
    # 如果输入已经是字典格式，说明是已格式化的用户信息，直接处理并返回
    if isinstance(query, dict):
        print("🎯 检测到已格式化的用户信息，跳过LLM提取步骤")
        info = query.copy()  # 复制字典以避免修改原始数据
        
        # 确保所有必需字段都存在
        required_fields = ['major', 'score', 'chinese', 'math', 'foreign_language',
                           'reselection_1', 'reselection_2', 'reselection_3',
                           'foreign', 'area', 'university', 'is_color_blindness']
        for field in required_fields:
            if field not in info:
                info[field] = None
        
        # 设置默认值
        if info.get('foreign') is None:
            info['foreign'] = 1
        if info.get('is_color_blindness') is None:
            info['is_color_blindness'] = 0
        
        # 处理副科信息，设置firstChoice和reselection（如果尚未设置）
        if 'firstChoice' not in info or 'reselection' not in info:
            firstChoice = None
            reselection = ["不限"]
            
            # 提取副科信息
            reselections = []
            for i in range(1, 4):
                field_name = f'reselection_{i}'
                if info.get(field_name) and isinstance(info[field_name], list) and len(info[field_name]) >= 1:
                    subject = info[field_name][0]  # 只取科目名称
                    reselections.append(subject)
            
            # 如果有副科信息
            if reselections:
                # 第一门副科存储到firstChoice（物理或历史）
                first_subject = reselections[0]
                if first_subject in ['物', '物理']:
                    firstChoice = '物'
                elif first_subject in ['史', '历史', '历']:
                    firstChoice = '史'
                else:
                    firstChoice = first_subject
                
                # 其他副科添加到reselection列表
                other_subjects = reselections[1:] if len(reselections) > 1 else []
                if other_subjects:
                    reselection.extend(other_subjects)
            
            info['firstChoice'] = firstChoice
            info['reselection'] = reselection
        
        print(f"✅ 已格式化用户信息处理完成: {info}")
        return info
    
    # 如果输入是字符串，使用原有的LLM提取逻辑
    if not isinstance(query, str):
        print(f"❌ 输入类型错误: {type(query)}，期望 str 或 dict")
        return None
    
    print("🤖 使用LLM提取用户信息...")
    # 创建提示词
    prompt = f"""请从以下用户输入中提取高考志愿填报所需的信息，并按照指定格式输出：

用户输入：{query}

请提取以下信息：
1. 意向专业（major）：提取专业名称，如"电子信息类"。如果有多个专业，可以返回数组格式，如["自动化", "机械类", "电气工程及其自动化"]
2. 不考虑的专业(not_major)：提取不考虑(不希望学、拒绝)的专业列表，如"电子信息类"。如果有多个专业，可以返回数组格式，如["自动化", "机械类", "电气工程及其自动化"]
3. 总分（score）：提取高考总分数字
4. 语文分数（chinese）：提取语文单科分数
5. 数学分数（math）：提取数学单科分数
6. 外语（foreign_language）：提取外语类型和分数，格式为["语种", 分数]，如["英", 100]
7. 副科1（reselection_1）：提取第一门副科类型和分数，格式为["科目", 分数]，如["物", 100]
8. 副科2（reselection_2）：提取第二门副科类型和分数，格式为["科目", 分数]，如["化", 100]
9. 副科3（reselection_3）：提取第三门副科类型和分数，格式为["科目", 分数]，如["生", 100]
10. 中外合作（foreign）：
   -如果明确说明不考虑中外合作办学或者用户输入说明家庭条件普通（来自普通家庭，不是富二代，来自小县城、农村等），取值0；
   -当用户明确要求学费不超过3万时，取值为1；
   -当用户明确要求学费3万-6万时，取值为2；
   -其他情况表明用户不考虑学费问题，取值为3。默认情况下取值3；
11. 报考批次（recruit）：本科B段（0）、本科提前批A段（1）、本科提前批B段（2）、本科A段国家专项（3）、本科A段地方专项（4）、本科高校专项（5）、区域均衡专项/省属预科（6）、专科提前批（7）、高职(专科)（8）。属于哪个批次就取哪个批次对应的数字。
12. 意向城市（city）：提取意向城市列表，要输出完整名称，如["北京市","成都市","浙江省"]，如果遇到沿海地区、新一线等说法，要输出具体的城市或者省份名。
13. 不考虑的城市(not_city)：提取不考虑(不希望去、拒绝)的城市列表，如["北京市","成都市","浙江省"]，如果遇到沿海地区、新一线等说法，要输出具体的城市或者省份名。
14. 意向学校（university）：提取意向学校列表，如["电子科技大学"]
15. 是否色盲（is_color_blindness）：如果提到色盲或色弱取值1，否则取值0
16. 是否考虑预科（matriculation）：如果明确说明考虑预科，取值1，默认情况下取值0（不考虑预科）
17. 是否考虑定向（target）：如果明确说明考虑定向，取值1，默认情况下取值0（不考虑定向）
18. 优先考虑维度（dimension）：如果明确说明优先考虑院校取值为1，如果明确说明优先考虑城市取值为2，其他情况下取值0，其中0=优先考虑专业，1=优先考虑院校，2=优先考虑城市
19. 选择民办院校还是公办院校（nature）：如果明确说明只考虑公办院校，取值为1；如果只考虑民办院校，取值为0；如果都考虑为None。默认为None

示例：
用户输入：我是四川的普通家庭考生，总分630分，语文100分，数学100分，英语100分，物理100分，化学100分，生物100分，想学电子信息类专业，想去川渝地区读书，不想去北上广等一线城市，特别想去电子科技大学。我想报考的批次是本科B段。

输出：
{{
    "major": "电子信息类",
    "not_major": None,
    "score": 630,
    "chinese": 100,
    "math": 100,
    "foreign_language": ["英", 100],
    "reselection_1": ["物", 100],
    "reselection_2": ["化", 100],
    "reselection_3": ["生", 100],
    "foreign": 0,
    "recruit": 0,   
    "city": ["四川省","重庆市"],
    "not_city": ["北京市","上海市","广东省"],
    "university": ["电子科技大学"],
    "is_color_blindness": 0,
    "matriculation": 0, 
    "target": 0,
    "dimension": 0,
    "nature": None,
}}

请严格按照以下JSON格式输出，不要添加任何其他内容：
{{
    "major": "专业名称" 或 ["专业1", "专业2", "专业3"],
    "not_major": "专业名称" 或 ["专业1", "专业2", "专业3"],
    "score": 总分数字,
    "chinese": 语文分数,
    "math": 数学分数,
    "foreign_language": ["语种", 分数],
    "reselection_1": ["科目", 分数],
    "reselection_2": ["科目", 分数],
    "reselection_3": ["科目", 分数],
    "foreign": 0或1,
    "recruit": 0或1或2或3或4或5或6或7或8
    "city": ["城市1", "城市2"],
    "not_city": ["城市1", "城市2"],
    "university": ["学校1", "学校2"],
    "is_color_blindness": 0或1,
    "matriculation": 0或1,
    "target": 0或1,
    "dimension": 0或1或2,
    "nature": 0或1或None
}}

注意：
- 如果某项信息不存在，请将对应字段设为null
- 副科按照物理、化学、生物、政治、历史、地理的顺序提取
- 确保提取的信息准确且完整
- 只输出标准JSON格式，不要添加任何其他文字或解释

"""

    try:
        # 使用大模型处理
        response = client_userInfo.chat.completions.create(
            model=USERINFO_MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个信息提取助手，专门从用户输入中提取高考志愿填报相关信息。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        # 从响应中提取JSON部分
        response_text = response.choices[0].message.content
        # 查找第一个 { 和最后一个 } 之间的内容
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("未找到有效的JSON格式")

        json_str = response_text[start:end]

        # 修复常见的JSON格式问题
        json_str = json_str.replace('None', 'null')  # Python None -> JSON null
        json_str = json_str.replace('True', 'true')  # Python True -> JSON true
        json_str = json_str.replace('False', 'false')  # Python False -> JSON false

        # 处理可能的中文冒号和逗号
        import re
        json_str = re.sub(r'：', ':', json_str)  # 中文冒号 -> 英文冒号
        json_str = re.sub(r'，', ',', json_str)  # 中文逗号 -> 英文逗号

        info = json.loads(json_str)

        # 验证必要字段
        if not isinstance(info, dict):
            raise ValueError("响应不是有效的字典格式")

        # 确保所有字段都存在
        required_fields = ['major', 'score', 'chinese', 'math', 'foreign_language',
                           'reselection_1', 'reselection_2', 'reselection_3',
                           'foreign', 'city', 'university', 'is_color_blindness']
        for field in required_fields:
            if field not in info:
                info[field] = None  # 这里保持None，因为Python代码中需要使用None

        # 特殊处理：foreign字段默认值为1（考虑中外合作），is_color_blindness字段默认值为0（不是色盲）
        if info.get('foreign') is None:
            info['foreign'] = 1
        if info.get('is_color_blindness') is None:
            info['is_color_blindness'] = 0

        # 处理副科信息，设置firstChoice和reselection
        firstChoice = None  # Python代码中使用None
        reselection = ["不限"]  # "不限"是必须的第一个字段

        # 提取副科信息
        reselections = []
        for i in range(1, 4):
            field_name = f'reselection_{i}'
            if info.get(field_name) and isinstance(info[field_name], list) and len(info[field_name]) >= 1:
                subject = info[field_name][0]  # 只取科目名称，不要分数
                reselections.append(subject)

        # 如果有副科信息
        if reselections:
            # 第一门副科存储到firstChoice（物理或历史）
            first_subject = reselections[0]
            if first_subject in ['物', '物理']:
                firstChoice = '物'
            elif first_subject in ['史', '历史', '历']:
                firstChoice = '史'
            else:
                firstChoice = first_subject  # 如果不是物理/历史，直接使用原值

            # 其他副科添加到reselection列表（"不限"后面）
            other_subjects = reselections[1:] if len(reselections) > 1 else []
            if other_subjects:
                reselection.extend(other_subjects)

        # 处理major字段 - 使用专业大类提取逻辑
        if info.get('major'):
            major_input = info['major']
            print(f"🔍 [DEBUG] 原始专业输入: {major_input}, 类型: {type(major_input)}")
            
            # 处理不同类型的专业输入
            if isinstance(major_input, list):
                # 如果已经是列表，直接使用
                print(f"✅ [DEBUG] 专业已经是列表格式: {major_input}")
                # 对每个专业进行标准化
                all_standardized_majors = []
                for major in major_input:
                    if isinstance(major, str):
                        standardized = extract_major_category(major)
                        
                        if standardized and isinstance(standardized, list):
                            all_standardized_majors.extend(standardized)
                        else:
                            # 如果标准化失败，保持原值
                            all_standardized_majors.append(major)
                
                # 去重并保持顺序
                seen = set()
                unique_majors = []
                for major in all_standardized_majors:
                    if major not in seen:
                        seen.add(major)
                        unique_majors.append(major)
                
                info['major'] = unique_majors
                print(f"✅ [DEBUG] 列表专业标准化结果: {unique_majors}")
                
            elif isinstance(major_input, str):
                # 如果是字符串，使用原有逻辑
                standardized_major_list = extract_major_category(major_input)
                if standardized_major_list and isinstance(standardized_major_list, list):
                    info['major'] = standardized_major_list
                    print(f"✅ [DEBUG] 字符串专业标准化结果: {standardized_major_list}")
                # 如果标准化失败，保持原值

        # 处理not_major字段 - 使用专业大类提取逻辑
        if info.get('not_major'):
            not_major_input = info['not_major']
            print(f"🔍 [DEBUG] 原始不考虑专业输入: {not_major_input}, 类型: {type(not_major_input)}")
            
            # 处理不同类型的专业输入
            if isinstance(not_major_input, list):
                # 如果已经是列表，直接使用
                print(f"✅ [DEBUG] 不考虑专业已经是列表格式: {not_major_input}")
                # 对每个专业进行标准化
                all_standardized_not_majors = []
                for major in not_major_input:
                    if isinstance(major, str):
                        standardized = extract_major_category(major)
                        
                        if standardized and isinstance(standardized, list):
                            all_standardized_not_majors.extend(standardized)
                        else:
                            # 如果标准化失败，保持原值
                            all_standardized_not_majors.append(major)
                
                # 去重并保持顺序
                seen = set()
                unique_not_majors = []
                for major in all_standardized_not_majors:
                    if major not in seen:
                        seen.add(major)
                        unique_not_majors.append(major)
                
                info['not_major'] = unique_not_majors
                print(f"✅ [DEBUG] 列表不考虑专业标准化结果: {unique_not_majors}")
                
            elif isinstance(not_major_input, str):
                # 如果是字符串，使用原有逻辑
                standardized_not_major_list = extract_major_category(not_major_input)
                if standardized_not_major_list and isinstance(standardized_not_major_list, list):
                    info['not_major'] = standardized_not_major_list
                    print(f"✅ [DEBUG] 字符串不考虑专业标准化结果: {standardized_not_major_list}")
                # 如果标准化失败，保持原值

        # 处理city字段 - 使用城市名称映射逻辑
        if info.get('city'):
            city_input = info['city']
            print(f"🔍 [DEBUG] 原始城市输入: {city_input}, 类型: {type(city_input)}")
            
            # 处理不同类型的城市输入
            if isinstance(city_input, list):
                # 如果已经是列表，对每个城市进行映射
                print(f"✅ [DEBUG] 城市已经是列表格式: {city_input}")
                all_mapped_cities = []
                for city in city_input:
                    if isinstance(city, str):
                        mapped = extract_city_name(city)
                        
                        if mapped and isinstance(mapped, list):
                            all_mapped_cities.extend(mapped)
                        else:
                            # 如果映射失败，保持原值
                            all_mapped_cities.append(city)
                
                # 去重并保持顺序
                seen = set()
                unique_cities = []
                for city in all_mapped_cities:
                    if city not in seen:
                        seen.add(city)
                        unique_cities.append(city)
                
                info['city'] = unique_cities
                print(f"✅ [DEBUG] 列表城市映射结果: {unique_cities}")
                
            elif isinstance(city_input, str):
                # 如果是字符串，使用城市映射逻辑
                mapped_city_list = extract_city_name(city_input)
                if mapped_city_list and isinstance(mapped_city_list, list):
                    info['city'] = mapped_city_list
                    print(f"✅ [DEBUG] 字符串城市映射结果: {mapped_city_list}")
                # 如果映射失败，保持原值

        # 处理not_city字段 - 使用城市名称映射逻辑
        if info.get('not_city'):
            not_city_input = info['not_city']
            print(f"🔍 [DEBUG] 原始不考虑城市输入: {not_city_input}, 类型: {type(not_city_input)}")
            
            # 处理不同类型的城市输入
            if isinstance(not_city_input, list):
                # 如果已经是列表，对每个城市进行映射
                print(f"✅ [DEBUG] 不考虑城市已经是列表格式: {not_city_input}")
                all_mapped_not_cities = []
                for city in not_city_input:
                    if isinstance(city, str):
                        mapped = extract_city_name(city)
                        
                        if mapped and isinstance(mapped, list):
                            all_mapped_not_cities.extend(mapped)
                        else:
                            # 如果映射失败，保持原值
                            all_mapped_not_cities.append(city)
                
                # 去重并保持顺序
                seen = set()
                unique_not_cities = []
                for city in all_mapped_not_cities:
                    if city not in seen:
                        seen.add(city)
                        unique_not_cities.append(city)
                
                info['not_city'] = unique_not_cities
                print(f"✅ [DEBUG] 列表不考虑城市映射结果: {unique_not_cities}")
                
            elif isinstance(not_city_input, str):
                # 如果是字符串，使用城市映射逻辑
                mapped_not_city_list = extract_city_name(not_city_input)
                if mapped_not_city_list and isinstance(mapped_not_city_list, list):
                    info['not_city'] = mapped_not_city_list
                    print(f"✅ [DEBUG] 字符串不考虑城市映射结果: {mapped_not_city_list}")
                # 如果映射失败，保持原值

        # 将处理后的变量添加到返回结果
        info['firstChoice'] = firstChoice
        info['reselection'] = reselection

        code=health_code(query)
        info['healthCheckupLimit'] = code

        # 修正dimension，若major不为空，则dimension设置为0
        print("dimension:",info.get('dimension'))
        if info.get('major'):
            info['dimension'] = 0
            
        return info

    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {str(e)}")
        if 'response' in locals():
            print(f"原始响应: {response.choices[0].message.content}")
        if 'json_str' in locals():
            print(f"处理后的JSON: {json_str}")
        return None
    except Exception as e:
        print(f"解析大模型响应时出错: {str(e)}")
        if 'response' in locals():
            print(f"原始响应: {response.choices[0].message.content}")
        return None


def health_code(user_input):
    prompt=f"""你需要根据用户的基本信息，从中提取出与身体健康有关的内容，判断用户是否属于哪个体检受限代码。
    要求：
    1.只输出一个或多个数字，例如"11"或"11,12"，不输出其他任何字符或解释或分析过程。
    2.如果用户身体健康，或者没有符合的体检受限代码，直接输出000。
    3.如果用户对应多个体检受限代码，则输出多个代码，用逗号分隔。例如"11,12"
    4.默认输出000
    
    体检受限代码：
    000：正常
    11：严重心脏病（经二级以上医院专科检查确定无需手术者除外），心肌病、高血压病。
    12：重症支气管扩张、哮喘、恶性肿瘤、慢性肾炎、尿毒症。
    13：严重的血液、内分泌及代谢系统疾病、风湿性疾病。
    14：重症或难治性癫痫或其他神经系统疾病；严重精神病未治愈、精神活性物质滥用和依赖。
    15：慢性肝炎病人并且肝功能不正常者（肝炎病原携带者但肝功能正常者除外）。
    16：结核病除下列情况外可以不予录取：（1）原发型肺结核、浸润性肺结核已硬结稳定；结核型胸膜炎已治愈或治愈后遗有胸膜肥厚者；（2）一切肺外结核（肾结核、骨结核、腹膜结核等等），血行性播散型肺结核治愈后一年以上未复发，经二级以上医院（或结核病防治所）专科检查无变化者；（3）淋巴腺结核已临床治愈无症状。
    21：轻度色觉异常（俗称色弱）
    22：色觉异常II度（俗称色盲）
    23-1：不能准确识别红、黄、绿、兰、紫各种颜色中任何一种颜色的导线、按键、信号灯、几何图形者
    23-2：不能准确在显示器上识别红、黄、绿、兰、紫各颜色中任何一种颜色的数码、字母者。
    24：裸眼视力任何一眼低于5.0者
    25：裸眼视力任何一眼低于4.8者
    31：主要脏器：肺、肝、肾、脾、胃肠等动过较大手术，功能恢复良好，或曾患有心肌炎、胃或十二指肠溃疡、慢性支气管炎、风温性关节炎等病史、甲状腺机能亢进已治愈一年
    32：先天性心脏病手术治愈，或房室间隔缺损分流量少、动脉导管未闭，返流血量少，经二级以上医院专科检查确定无需手术者不宜就读
    33：肢体残疾，生活能自理
    34：屈光不正（近视眼或远视眼，下同）任何一眼矫正到4.8镜片度数大于400度着
    35：任何一眼矫正到4.8镜片度数大于800度者
    36：一眼失明，另一眼矫正到4.8镜片度数大于400度者
    37：两耳听力均在3米以内，或一耳听力在5米另一耳全聋的。  
    38：嗅觉迟钝、口吃、步态异常、驼背，面部有疤痕、血管瘤、黑色素痣、白癜风者
    39：斜视、嗅觉迟钝、口吃

    
    用户基本信息：{user_input}
    
"""
    try:
        response = client_userInfo.chat.completions.create(
            model=USERINFO_MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        response_text = response.choices[0].message.content.strip()
        return response_text

    except Exception as e:
        print(f"体检代码提取失败: {str(e)}")
        return None



# 用户信息字段定义（用于其他模块参考）
USER_INFO_FIELDS = {
    'major': '意向专业（官方规定大类列表）',
    'score': '总分',
    'chinese': '语文分数',
    'math': '数学分数',
    'foreign_language': '外语类型和分数',
    'reselection_1': '副科1类型和分数',
    'reselection_2': '副科2类型和分数',
    'reselection_3': '副科3类型和分数',
    'foreign': '中外合作（1=考虑[默认]，0=不考虑）',
    'city': '意向城市列表',
    'university': '意向学校列表',
    'is_color_blindness': '是否色盲（0=否[默认]，1=是）',
    'firstChoice': '第一选科（物/史）',
    'reselection': '其他副科列表（"不限"为必须首位）',
    'dimension': '优先考虑维度（0=优先考虑专业，1=优先考虑院校，2=优先考虑城市）'
}



def merge_additional_majors(original_query: str, additional_input: str, expansion_type: str = "manual", session_id: str = "student_001") -> str:
    """
    合并原始查询和额外专业信息，生成新的查询字符串

    Args:
        original_query (str): 原始用户查询
        additional_input (str): 额外的专业信息输入
        expansion_type (str): 扩展类型，"manual"表示手动扩展，"auto"表示自动扩展
        session_id (str): 当前会话的唯一标识id，用于日志记录和调试

    Returns:
        str: 合并后的新查询字符串
    """
    if expansion_type == "manual":
        # 手动扩展：直接合并用户补充的专业信息
        merged_query = f"{original_query}，补充的意向专业或意向城市：{additional_input}"
    else:
        # 自动扩展：将相关专业作为补充信息
        merged_query = f"{original_query}，补充的意向专业或意向城市：{additional_input}"
    
    print(f"🔄 专业扩展合并 (session_id: {session_id}):")
    print(f"   原始查询: {original_query}")
    print(f"   补充信息: {additional_input}")
    print(f"   扩展类型: {expansion_type}")
    print(f"   合并结果: {merged_query}")
    
    return merged_query


# 如果直接运行此文件，则执行测试
if __name__ == "__main__":
    print("🚀 开始测试用户信息提取模块")

    # print("测试专业大类提取功能")
    # test_major_category_extraction()
    # print()


    query = "我是一名四川省历史组考生，汉族，无民族照顾分，无专项可以报考，体检正常，选科:历史+政治+地理，高考成绩594分，大学毕业后优先考虑考研，其次考虑好就业的专业，中外合作不考虑，请给一份完整的本科B段志愿方案"
    print("用户输入:",query)
    result = format_user_info_with_llm(query)
    print(result)

    # health_code(query)

    # print("测试完整的用户信息提取功能")
    # test_format_user_info_function()


