import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict
from userInfo import extract_major_category

load_dotenv()


def get_lowest_probability_schools(data,num):
    """
    从data中提取所有专业，按照录取概率区间筛选，返回25条学校数据
    筛选规则：50-60%区间选10个，60-70%选5个，70-80%选5个，80-90%选5个，90%以上不选
    当某概率区间个数不足时，从下一概率区间补齐，直到选满25个
    保持与原数据相同的JSON格式
    """
    # 1. 提取所有专业数据并展平
    all_majors = []

    # 遍历每个学校的专业列表
    for school_name, majors in data['answer'].items():
        for major in majors:
            # 确保专业信息中包含学校名称
            major_info = major.copy()
            major_info['院校名称'] = school_name
            all_majors.append(major_info)

    # 2. 按照录取概率排序（从小到大）
    # 注意：录取概率是字符串，需要转换为整数进行排序
    sorted_majors = sorted(all_majors, key=lambda x: int(x.get('录取概率', '0')))

    # 3. 按概率区间分组
    ranges = {
        '50-60': [],  # 50 <= prob < 60
        '60-70': [],  # 60 <= prob < 70
        '70-80': [],  # 70 <= prob < 80
        '80-90': [],  # 80 <= prob < 90
        '90+': []     # prob >= 90
    }
    
    for major in sorted_majors:
        prob = int(major.get('录取概率', '0'))
        if 50 <= prob < 60:
            ranges['50-60'].append(major)
        elif 60 <= prob < 70:
            ranges['60-70'].append(major)
        elif 70 <= prob < 80:
            ranges['70-80'].append(major)
        elif 80 <= prob < 90:
            ranges['80-90'].append(major)
        elif prob >= 90:
            ranges['90+'].append(major)

    # 4. 按区间目标数量筛选，不足时从下一区间补齐
    target_counts = {
        '50-60': 10,
        '60-70': 5,
        '70-80': 5,
        '80-90': 5,
        '90+': 0
    }
    
    selected_majors = []
    total_target = 25  # 总共需要25个
    deficit = 0  # 记录需要从下一区间补齐的数量
    
    # 按顺序处理每个区间
    range_order = ['50-60', '60-70', '70-80', '80-90']
    
    for range_key in range_order:
        available = ranges[range_key]
        target = target_counts[range_key]
        
        # 当前区间需要选择的数量 = 目标数量 + 上一区间不足的数量
        needed_from_this_range = target + deficit
        
        # 实际能选择的数量（不超过可用数量，也不超过剩余需要的总数）
        remaining_total = total_target - len(selected_majors)
        can_select = min(needed_from_this_range, len(available), remaining_total)
        
        if can_select > 0:
            selected_majors.extend(available[:can_select])
        
        # 计算当前区间还缺多少（用于下一区间补齐）
        deficit = needed_from_this_range - can_select
        
        # 如果已经选满25个，停止
        if len(selected_majors) >= total_target:
            break

    # 5. 按学校重新组织数据结构，保持与原格式一致
    result = defaultdict(list)
    for major in selected_majors:
        school_name = major['院校名称']
        # 移除添加到major中的院校名称，因为原格式中每个专业已经包含院校名称
        major_without_school_name = {}
        for key, value in major.items():
            if key != '院校名称':
                major_without_school_name[key] = value

        result[school_name].append(major_without_school_name)

    # 6. 创建返回的JSON结构
    result_data = {
        'question': data['question'],
        'answer': dict(result)
    }

    return result_data

def get_keywords(user_inputs):
    """
    从用户的输入语句中提取符合接口要求的关键词，遇到某些缩写时能够识别并扩展为正确的参数

    参数:
        user_inputs (str): 用户的自然语言查询输入

    返回:
        tuple: (score , universityName ,majorSecondDesc, provinceName, firstChoice, has_province_input)
            score (int): 提取出的分数，没有时返回None
            universityName (str): 提取出的院校名称，没有时返回None
            majorSecondDesc (str): 提取出的专业名称，没有时返回None
            provinceName (str): 提取出的省市名称，默认返回"四川省"
            firstChoice (str): 选科类别，物理类传"物"，历史类传"史"，无法判断时为None
            has_province_input (bool): 用户是否显式输入了生源地
    """
    majorSecondDesc = extract_major_category(user_inputs)
    majorSecondDesc = ",".join(majorSecondDesc) if majorSecondDesc else None
    client = OpenAI(
        base_url=os.getenv("THINKING_BASE_URL"),
        api_key=os.getenv("THINKING_API_KEY")
    )

    prompt = f"""
    你是一个专业的关键词提取器。请从下面的用户输入中提取出相应的关键词。

    要求：
    1. 从用户的自然语言输入中提取出分数、院校名称、专业名称、省市名称这四个关键词
    2. 对于提取出的分数进行处理，确保是int类型
    3. **当且仅当用户明确说明了院校名后**：将院校名称全部提取出来（多个的话用,隔开），检查院校名称是否是某个学校的简称或者别称，是的话将其扩展为正确的大学全称;如果没有检测到院校名称则返回"无"
    4. 识别省市名称：支持省级（如“四川省”）或市级（如“成都市”）命名；如果用户没有明确指定省市名称则返回"无"
    5. 识别用户的选科类别：如果是物理类考生，请返回"物"；如果是历史类考生，请返回"史"；无法判断时返回"无"
    6. 返回格式：严格按照以下格式返回，不要包含其他内容
       分数: 提取出的分数
       院校名称: 提取出的院校名称，如果是多个则用","隔开，没有则返回"无"
       省市名称: 提取出的省市名称，没有则返回"无"
       选科类别: 物/史/无

    用户输入：{user_inputs}
    """

    try:
        # 构建请求
        response = client.chat.completions.create(
            model=os.environ.get("THINKING_MODEL_NAME"),
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的关键词提取助手，能够准确地将自然语言中的关键词提取出来，遇到关键词的缩写也能够进行拓展处理。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=500
        )

        if response.choices:
            content = response.choices[0].message.content.strip()
            lines = content.split("\n")
            score = None
            universityName = None
            provinceName = "四川省"
            has_province_input = False
            firstChoice = None
            for line in lines:
                if line.startswith("分数:"):
                    score = line.split(":", 1)[1].strip()
                elif line.startswith("院校名称:"):
                    university = line.split(":", 1)[1].strip()
                    universityName = university if university != "无" else None
                elif line.startswith("省市名称:"):
                    pn = line.split(":", 1)[1].strip()
                    has_province_input = bool(pn and pn != "无")
                    provinceName = pn if has_province_input else "四川省"
                elif line.startswith("选科类别:"):
                    fc = line.split(":", 1)[1].strip()
                    # 只接受"物"或"史"，其他情况按无法判断处理
                    firstChoice = fc if fc in {"物", "史"} else None
            print(f"score: {score}, universityName: {universityName}, majorSecondDesc: {majorSecondDesc}, provinceName: {provinceName}, firstChoice: {firstChoice}")
            return score, universityName, majorSecondDesc, provinceName, firstChoice, has_province_input
        return None, None, None, "四川省", None, False

    except Exception as e:
        print(f"提取关键词出错: {e}")
        return None, None, None, "四川省", None, False

def special_question(input_text):
    """
    处理两类问题：
    1. 指定院校：查询某分数报考某院校某专业的录取概率
    2. 未指定院校：根据分数和专业推荐符合条件的院校
    
    参数:
        input_text (str): 用户的自然语言查询输入
        
    返回:
        str: 格式化的查询结果字符串
    """
    score, universityName, majorSecondDesc, provinceName, firstChoice, has_province_input = get_keywords(input_text)
    notice_prefix = ""
    if not has_province_input:
        notice_prefix = "注意：用户没有输入生源地，因此系统默认为四川生源地考生，返回的结果也仅适用于四川生源地考生，请告知用户\n"

    num = 25
    
    # 检查必要参数
    if not score:
        return "抱歉，无法从您的输入中提取到分数信息。请确保输入包含您的分数。"
    
    if not majorSecondDesc:
        return "抱歉，无法从您的输入中提取到专业信息。请确保输入包含专业名称。"

    if universityName is not None:
        # 情况1：指定了院校，查询录取概率
        url = "https://rest.nextgoo.cn/nextgoo/v1/aimodel/getScoreUniversityMajorRate"
        payload = json.dumps({
            "provinceName": provinceName,
            "score": score,
            "universityName": universityName,
            "majorSecondDesc": majorSecondDesc,
            "firstChoice": firstChoice if firstChoice else ""
        })
        headers = {
            'token': '{{token}}',
            'Content-Type': 'application/json'
        }
        try:
            response = requests.request("GET", url, headers=headers, data=payload)
            if response.status_code == 200:
                result = response.json()
                output = result.get('data', {})
                #print("原始返回数据：",result)
                # 处理并返回 question + answer 的 JSON（去掉最外层 msg/code/data）
                if isinstance(output, dict):
                    # 仅保留 question 和 answer 结构
                    output = {
                        "question": output.get("question"),
                        "answer": output.get("answer", {})
                    }
                    # 将录取概率为 -1 或 -1% 的项标记为“新增专业”
                    answer = output.get("answer", {})
                    if isinstance(answer, dict):
                        for _, majors in answer.items():
                            if isinstance(majors, list):
                                for major in majors:
                                    prob = str(major.get("录取概率", ""))
                                    if prob in ["-1", "-1%"]:
                                        major["录取概率"] = "新增专业"
                                    elif prob and not prob.endswith("%"):
                                        major["录取概率"] = f"{prob}%"
                # 预留字符串接口，供上层按需拼接提示信息
                formatted_result = json.dumps(output, ensure_ascii=False, indent=2)
                formatted_result = notice_prefix + formatted_result
                return formatted_result
            else:
                return f"查询失败，API返回状态码：{response.status_code}"
        except Exception as e:
            return f"查询过程中出错：{str(e)}"
    else:
        # 情况2：未指定院校，推荐院校
        url = "https://rest.nextgoo.cn/nextgoo/v1/aimodel/getScoreMajorUniversity"
        payload = json.dumps({
            "provinceName": provinceName,
            "score": score,
            "majorSecondDesc": majorSecondDesc,
            "firstChoice": firstChoice if firstChoice else ""
        })
        headers = {
            'token': '{{token}}',
            'Content-Type': 'application/json'
        }
        try:
            response = requests.request("GET", url, headers=headers, data=payload)
            if response.status_code == 200:
                result = response.json()
                output = result.get('data', '')
                if output and isinstance(output, dict):
                    output = get_lowest_probability_schools(output, num)
                    
                    # 格式化输出结果
                    formatted_result = f"以下是根据用户请求得到的院校及专业推荐结果（按录取概率从低到高排序）：\n\n"
                    for school_name, majors in output.get('answer', {}).items():
                        formatted_result += f"**{school_name}**\n"
                        for major in majors:
                            prob = str(major.get('录取概率', 'N/A'))
                            if prob in ['-1', '-1%']:
                                prob_text = "新增专业"
                            else:
                                prob_text = f"{prob}%"
                            formatted_result += f"  - 专业组：{major.get('专业组', 'N/A')}\n"
                            formatted_result += f"    专业名称：{major.get('专业名称', 'N/A')}\n"
                            formatted_result += f"    录取概率：{prob_text}\n"
                            formatted_result += f"    选科要求：{major.get('选科要求', 'N/A')}\n"
                            formatted_result += f"    招生批次：{major.get('招生批次', 'N/A')}\n\n"
                    formatted_result = notice_prefix + formatted_result
                    return formatted_result
                else:
                    return "抱歉，未找到符合条件的院校推荐。"
            else:
                return f"查询失败，API返回状态码：{response.status_code}"
        except Exception as e:
            return f"查询过程中出错：{str(e)}"



if __name__ == "__main__":
    #示例1: 我高考500分，想读成都大学的计算机专业，有多大机会？
    #示例2: 我高考500分，想读计算机专业，有哪些学校可以推荐？
    input_text = input("请输入你的问题：")
    res = special_question(input_text)
    print(res)
   

