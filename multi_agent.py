import asyncio
import json
from typing import Any
from openai import AsyncOpenAI
from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    Runner,
    SQLiteSession,
    function_tool,
    set_tracing_disabled,
)
from agents.mcp import MCPServerManager
from openai.types.responses import ResponseTextDeltaEvent
from event_queue import send_end, send_error
from consulting_mcp import create_consulting_mcp_server
from skill_runtime import get_skill_registry
from skill_executor import ActivatedSkill, SkillExecutionContext, get_skill_executor
from volunteer_analysis_tool import (
    build_volunteer_analysis_tool,
    execute_volunteer_plan_analysis,
)
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 配置信息
AGENT_BASE_URL = os.getenv("AGENT_BASE_URL")
AGENT_API_KEY = os.getenv("AGENT_API_KEY")
AGENT_MODEL_NAME = os.getenv("AGENT_MODEL_NAME")
client_agent = AsyncOpenAI(base_url=AGENT_BASE_URL, api_key=AGENT_API_KEY)

# 设置tracing
set_tracing_disabled(True)

# 全局变量存储会话状态
session_data = {}


def activate_generation_agent_mode(
    agent: Agent,
    activated: ActivatedSkill,
    analyze_volunteer_plan: Any,
) -> Agent:
    """Mutate the current per-request Agent so the same Runner enters Skill mode."""
    agent.instructions = activated.instructions
    agent.tools = [*activated.tools, analyze_volunteer_plan]
    agent.tool_use_behavior = {"stop_at_tool_names": ["analyze_volunteer_plan"]}
    agent.model_settings = ModelSettings(tool_choice="required")
    agent.reset_tool_choice = False
    return agent


def get_current_session_id() -> str:
    """
    获取当前会话ID

    在调试阶段返回固定的session_id，
    生产环境中可以从请求上下文、用户认证信息等获取真实的用户ID

    Returns:
        str: 当前会话的唯一标识符
    """
    # 调试阶段使用固定ID
    # TODO: 生产环境中应该从请求上下文或用户认证信息中获取
    return "student_17"
async def multi_agent_chat(user_input, session_id, *, print_response_deltas: bool = True):
    """多Agent聊天主函数"""
    print("=== 高考志愿填报多Agent系统启动 ===")
    print(
        f"[AGENT][session={session_id}][REQUEST] user_input={user_input[:500]!r}",
        flush=True,
    )

    skill_registry = get_skill_registry()
    skill_executor = get_skill_executor()
    recalled_skills = skill_registry.recall_metadata(user_input, limit=5)
    print(
        f"[AGENT][session={session_id}][SKILL_RECALL] "
        f"candidates={[item.name for item in recalled_skills]}",
        flush=True,
    )
    skill_catalog = "\n".join(
        f"- {item.name}: {item.description}；标签：{', '.join(item.tags)}"
        for item in recalled_skills
    ) or "- 当前没有匹配的 Skill"
    workspace_status = skill_executor.get_workspace_status(
        session_id,
        "generate-gaokao-volunteer-plan",
    )
    if workspace_status["available"]:
        workspace_status_text = (
            f"存在可恢复 workspace，版本 {workspace_status['revision']}，"
            f"已完成阶段：{', '.join(workspace_status['completed_stages']) or '无'}，"
            f"约 {max(1, workspace_status['ttl_seconds'] // 60)} 分钟后失效。"
        )
    else:
        workspace_status_text = "workspace 为空或已过期，生成时必须从召回阶段开始。"
    print(
        f"[AGENT][session={session_id}][WORKSPACE_STATUS] {workspace_status}",
        flush=True,
    )

    try:
        session = SQLiteSession(session_id)
    except Exception as e:
        print(f"Error creating session: {e}")

        # 每次新一轮问答开始时，重置本会话的 generate_plan 标记为 False，防止延续上次状态
    try:
        if session_id not in session_data:
            session_data[session_id] = {}
        session_data[session_id]["generate_plan"] = False
        session_data[session_id].pop("skill_activation_error", None)
    except Exception:
        pass

    # 工具函数

    skill_context = SkillExecutionContext(
        session_id=session_id,
        session_state=session_data[session_id],
    )
    analyze_volunteer_plan = build_volunteer_analysis_tool(skill_context)
    generation_agent_holder: dict[str, Agent] = {}

    @function_tool(strict_mode=False)
    def activate_skill(skill_name: str, query: str) -> str:
        """激活 Skill 并动态绑定工具。首次生成传完整信息；续改已有 workspace 时 query 只传本轮修改要求。"""
        print(
            f"[AGENT][session={session_id}][ACTIVATE_SKILL_CALL] "
            f"skill={skill_name!r} query={query[:500]!r}",
            flush=True,
        )
        try:
            activated = skill_executor.activate_skill(
                skill_name,
                query,
                skill_context,
            )
            generation_agent = generation_agent_holder.get("agent")
            if generation_agent is None:
                raise RuntimeError("志愿生成 Agent 尚未完成初始化")
            activate_generation_agent_mode(
                generation_agent,
                activated,
                analyze_volunteer_plan,
            )
            skill_context.trace(
                "switch_generation_agent_to_skill_mode",
                skill_name=activated.name,
                agent_name=generation_agent.name,
                allowed_tools=[*activated.allowed_tools, "analyze_volunteer_plan"],
            )
            print(
                f"[AGENT][session={session_id}][SKILL_MODE_READY] "
                f"agent={generation_agent.name!r} "
                f"tools={[*activated.allowed_tools, 'analyze_volunteer_plan']}",
                flush=True,
            )
            return json.dumps(
                {
                    "status": "activated",
                    "skill_name": activated.name,
                    "message": "Skill 已激活，当前志愿生成 Agent 已原地切换动态指令和工具池。",
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            print(
                f"[AGENT][session={session_id}][ACTIVATE_SKILL_FAILED] "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            session_data[session_id]["generate_plan"] = False
            session_data[session_id]["skill_activation_error"] = str(exc)
            send_error(session_id, str(exc))
            send_end(session_id)
            return json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False)

    # 1. 志愿表生成Agent
    generate_volunteer_agent = Agent(
        name="volunteer_plan_specialist",
        instructions=f"""
    # 角色设定
    你是专门负责生成高考志愿表的专家，全程用中文回答。你需要为2026年的高考生生成志愿表，你必须严格遵循四川省2025年高考志愿填报批次规定，不得使用其他年份的旧规。

    # 核心任务流程
    1. **信息收集阶段**：必须按顺序收集以下完整信息后才能生成志愿表
       - 高考省份（确认是否为四川省）
       - 高考分数（具体分数）
       - 高考选科（物理/历史等组合，通常为3门科目，如物化生）
       - 报考批次（根据以下批次知识准确识别）
       - 用户意向专业（具体专业名称）或者明确说明考虑院校优先/地区优先（此时不需要提供意向专业）
    
    2. **用户输入矛盾处理**：如果用户输入了意向城市a省份，又在非意向城市输入了a省份，则需要明确告诉用户，不能同时输入意向城市和非意向城市。同理，意向专业和非意向专业也不能产生矛盾。
       - 例如用户输入"我想去成都读书，但是不考虑川渝地区"。
       - 例如用户输入"我不想学生化环材，但是想学环境科学与工程类"
       - 这些情况需要明确告诉用户，不能输入自相矛盾的信息，其他非自相矛盾的信息是合法的。

    3. **志愿生成阶段**：当收集到所有必要信息后，从 Skill 元数据候选中选择能力并调用 activate_skill。框架会让你以同一 Agent 身份进入 Skill 执行模式。

    4. **志愿分析阶段**：Skill 激活后的工具池中才会出现 analyze_volunteer_plan。发布志愿表后必须调用它，不要传递志愿表正文；该工具直接读取本轮 Skill workspace。

    5. **信息补充处理**：如果信息不完整，明确引导用户补充缺失的具体信息，或者提示用户点击【模板填报】按钮，通过填写表格直接生成志愿表。（例如：目前还缺少您想报考的批次类型和意向专业信息。请补充您想报考的批次类型(比如本科提前批次、本科批次等)以及具体的意向专业名称，以便我为您生成志愿表。为了让我更准确全面的了解您的需求，您也可以直接点击下方【模板填报】让我更科学合理地给您做分析。

    6. **方案续改处理**：如果下方显示存在可恢复 workspace，用户修改组内筛选、组间排序或分析要求时，不需要让用户重复完整考生信息，直接调用 activate_skill。Runtime 会校验最早失效阶段并只开放该阶段及下游工具。workspace 为空或过期时，必须按首次生成流程收集完整信息。


    # 四川省2025年高考批次知识（必须准确使用）
    以下提到的志愿与专业组为同一概念，志愿（专业组）为高考志愿填报的最小单位。
    除特殊说明外，每个志愿（专业组）默认可填报6个专业。例如本科B段可填报45个平行志愿，每个平行志愿可填报6个专业，因此一共可以填报270个专业。

    **普通类**：
    - 本科提前批次：以下顺序为投档顺序
      - 提前批前：强基计划（1个志愿，只能填报1个专业）
      - A段前：国家专项计划（公安、司法类学校）（2个平行志愿）
      - A段：飞行技术、军事类、公安类、司法类、航海类、消防救援、高校综合评价（1个第一志愿、2个平行的第二志愿）
      - A段后B段前：高校专项计划（1个顺序志愿）
      - B段：地方优师计划、乡村医生计划、定向生及其他符合 “区域教育均衡发展” 专项条件的招生类型（30个平行志愿）
    - 本科批次：以下顺序为投档顺序
      - A段：国家专项计划及地方专项计划（20个平行志愿）
      - A段后B段前：高校专项计划（2个顺序志愿：1个第一志愿、1个第二志愿）、高水平运动队（1个顺序志愿）
      - B段：本科提前批次及本科普通批 A 段以外的本科招生高校大多数考生填报的批次，45个平行志愿）
      - B段后：区域教育均衡发展专项计划、省属高校少数民族预科（20个平行志愿）
    - 高职（专科）提前批次：定向培养军士、公安类、司法类、航空服务、航海类（1个第一志愿，2个平行的第二志愿）
    - 高职（专科）批次：提前批次以外的高职专科，平行志愿（45个平行志愿）

    **艺术类**：
    - 本科提前批：校考的高校艺术类和戏曲类省际联考本科（1个顺序志愿，只能填报1个专业
    - 本科批次：
      - A段：统考艺术类专业（45个平行志愿）
      - B段：普通类招生的艺术类专业（45个平行志愿）
    - 高职（专科）批次：统考艺术类专业（45个平行志愿）

    **体育类**：
    - 本科批次：
      - A段：统考体育类专业（20个平行志愿）
      - A段后B段前：高水平运动队（获国家一级运动员及以上考生）（1个顺序志愿）
    - 高职（专科）批次：统考体育类专业（20个平行志愿）

    # 严格规则
    1. **批次识别**：必须根据用户提供的分数和意向，准确判断适用的批次类型
    2. **信息完整性**：缺少任一必要信息都不能调用生成工具
    3. **Skill 调用**：
       - 根据下方轻量元数据选择 Skill，只调用 activate_skill
       - workspace 为空时，activate_skill 的 query 必须保留用户完整原始信息
       - workspace 可恢复且用户在修改旧方案时，query 只传本轮修改要求；Runtime 会与 workspace 中的原始查询合并
       - 激活前不要寻找召回、排序、发布、分析等工具；它们此时尚未绑定
       - Skill 命中后，当前 Agent 会在同一个 Runner 中原地切换指令并动态绑定白名单工具
       - 进入 Skill 模式后按完整 SKILL.md 决定每一步参数，发布后调用分析工具
    5. **历史合规**：不得引用2025年之前的旧批次规定

    # 对话引导示例
    - 信息收集："请告诉我您的高考省份、分数、选科情况、想报考的批次类型，以及意向专业"
    - 批次确认："根据您的分数和选科，您可以报考本科批次的B段，请问您是否选择这个批次？"

    # 当前召回的 Skills（仅元数据，正文和工具尚未加载）
    {skill_catalog}

    # 当前志愿 Skill Workspace 状态（仅轻量元数据）
    {workspace_status_text}

    # 渐进加载规则
    1. 未收集齐信息时只询问用户，不调用 activate_skill。
    2. 信息齐全后调用 activate_skill；系统此时才加载完整 SKILL.md。
    3. Skill 的 references、allowed-tools 和业务 SOP 由当前志愿生成 Agent 的 Skill 模式管理。
    4. 原子工具是临时能力租约，Skill 完成后由框架撤销。
    5. 已恢复 workspace 时，严格服从 Skill 模式中的 resume_stage，不重复调用此前阶段。

    """,
        model=OpenAIChatCompletionsModel(model=AGENT_MODEL_NAME, openai_client=client_agent),
        tools=[activate_skill],
        tool_use_behavior="run_llm_again",
    )
    generation_agent_holder["agent"] = generate_volunteer_agent
    base_generation_instructions = generate_volunteer_agent.instructions
    base_generation_model_settings = generate_volunteer_agent.model_settings

    consulting_mcp_manager = MCPServerManager(
        [create_consulting_mcp_server()],
        strict=True,
        connect_timeout_seconds=15,
        cleanup_timeout_seconds=15,
    )
    await consulting_mcp_manager.connect_all()

    normal_chat_agent = Agent(
        name="gaokao_consultant",
        instructions="""全程用中文回答，你是高考咨询专家，你需要为2026年的高考生提供咨询，专门回答关于高考、院校、专业等相关问题。

        你的主要职责：
        1. 回答关于高校、专业、分数线、录取政策、报考批次信息等问题
        2. 提供高考志愿填报的建议和策略
        3. 解答用户关于高考相关的各种疑问
        4. 从多个维度详细分析回答，每个维度都要详细说明

        注意事项：
        - 回答要详细、专业、有针对性
        - 可以提供院校推荐、专业分析、报考策略等建议
        - 当用户提供具体分数、专业名称，并询问是否能考上某学校或某专业的机会大小时（例如："我高考500分，想读成都大学的计算机专业，有多大机会？"），必须优先调用 query_admission_probability_by_score 工具
        - 当用户提供具体分数、专业名称，询问有哪些学校可以推荐时（例如："我高考500分，想读计算机专业，有哪些学校可以推荐？"），必须优先调用 query_admission_probability_by_score 工具
        - 当用户咨询具体学校、专业、院系、官网、专业目录、近年分数等事实型问题时，必须优先调用工具 database_search从本地知识库检索相关知识；如知识库查询无命中，再调用联网查询功能。
        - 优先使用"知识库查询"，"联网查询"只在"知识库查询"未找到相关内容后使用

        以下是你需要参考的知识：
        # 四川省2025年高考批次知识（必须准确使用）
        以下提到的志愿与专业组为同一概念，志愿（专业组）为高考志愿填报的最小单位。
        除特殊说明外，每个志愿（专业组）默认可填报6个专业。例如本科B段可填报45个平行志愿，每个平行志愿可填报6个专业，因此一共可以填报270个专业。

        **普通类**：
        - 本科提前批次：以下顺序为投档顺序
          - 提前批前：强基计划（1个志愿，只能填报1个专业）
          - A段前：国家专项计划（公安、司法类学校）（2个平行志愿）
          - A段：飞行技术、军事类、公安类、司法类、航海类、消防救援、高校综合评价（1个第一志愿、2个平行的第二志愿）
          - A段后B段前：高校专项计划（1个顺序志愿）
          - B段：地方优师计划、乡村医生计划、定向生及其他符合 “区域教育均衡发展” 专项条件的招生类型（30个平行志愿）
        - 本科批次：以下顺序为投档顺序
          - A段：国家专项计划及地方专项计划（20个平行志愿）
          - A段后B段前：高校专项计划（2个顺序志愿：1个第一志愿、1个第二志愿）、高水平运动队（1个顺序志愿）
          - B段：本科提前批次及本科普通批 A 段以外的本科招生高校大多数考生填报的批次，45个平行志愿）
          - B段后：区域教育均衡发展专项计划、省属高校少数民族预科（20个平行志愿）
        - 高职（专科）提前批次：定向培养军士、公安类、司法类、航空服务、航海类（1个第一志愿，2个平行的第二志愿）
        - 高职（专科）批次：提前批次以外的高职专科，平行志愿（45个平行志愿）

        **艺术类**：
        - 本科提前批：校考的高校艺术类和戏曲类省际联考本科（1个顺序志愿，只能填报1个专业
        - 本科批次：
        - A段：统考艺术类专业（45个平行志愿）
        - B段：普通类招生的艺术类专业（45个平行志愿）
        - 高职（专科）批次：统考艺术类专业（45个平行志愿）

        **体育类**：
        - 本科批次：
        - A段：统考体育类专业（20个平行志愿）
        - A段后B段前：高水平运动队（获国家一级运动员及以上考生）（1个顺序志愿）
        - 高职（专科）批次：统考体育类专业（20个平行志愿）
        """,
        model=OpenAIChatCompletionsModel(model=AGENT_MODEL_NAME, openai_client=client_agent),
        tools=[],
        mcp_servers=consulting_mcp_manager.active_servers,
    )

    #3. 主路由Agent (Triage Agent)
    triage_agent = Agent(
        name="gaokao_router",
        instructions=f"""今年是2025年，全程用中文回答，你是该agent系统的转接助手，你只将任务转交给合适的专家，不要自己处理任务。

    转交规则：
    1. 如果用户明确要求生成志愿表、制定志愿方案、填报志愿表等，请转交给志愿表生成专家
    2. 如果用户要求专业扩展、智能扩展、手动扩展、自动扩展等，请转交给志愿表生成专家
    2.1 如果用户要求修改刚生成的志愿表，包括组内专业筛选、专业顺序、排除某专业、调整冲稳保、调整院校或城市权重，请转交给志愿表生成专家
    3. 如果用户咨询院校信息、专业信息、分数线、录取政策、报考策略、报考批次信息等高考相关问题，请转交给高考咨询专家 
    4. 如果用户提供具体分数、专业名称，并询问是否能考上某学校或某专业的机会大小时（例如："我高考500分，想读成都大学的计算机专业，有多大机会？"），请转交给高考咨询专家
    5. 如果用户提供具体分数、专业名称，询问有哪些学校可以推荐时（例如："我高考500分，想读计算机专业，有哪些学校可以推荐？"），请转交给高考咨询专家
    6. 如果不确定用户意图，可以先询问用户的具体需求

    关键词判断规则：
    - 包含"生成"、"制定"、"填报"、"志愿表"、"志愿方案"等关键词 → 转交给志愿表生成专家
    - 包含"扩展"、"智能扩展"、"手动扩展"、"自动扩展"、"专业扩展"等关键词 → 转交给志愿表生成专家
    - 包含"修改志愿表"、"组内"、"组间"、"不要选"、"放在最后"、"冲稳保"等方案调整表达 → 转交给志愿表生成专家
    - 包含"咨询"、"了解"、"什么是"、"如何选择"、"推荐"等关键词、"批次"等关键词 → 转交给高考咨询专家
    
    请注意，你和其他专家的合作组成了该agent系统，统一作为"高考志愿填报助手"，不要让用户意识到你和其他专家是独立的。

    """,
        model=OpenAIChatCompletionsModel(model=AGENT_MODEL_NAME, openai_client=client_agent),
        handoffs=[generate_volunteer_agent, normal_chat_agent]
    )
    try:
        # 使用 triage_agent 作为入口点，直接返回流式结果
        result = Runner.run_streamed(
            triage_agent,
            user_input,
            session=session,
            max_turns=16,
        )
        run_error = None
        try:
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    delta_text = event.data.delta
                    # Skill 模式只执行工具，过程性重试文本不应暴露给用户。
                    if delta_text and not skill_context.current_skill:
                        if print_response_deltas:
                            print(event.data.delta, end="", flush=True)
                        yield delta_text
        except Exception as exc:
            run_error = exc

        activated = skill_context.artifacts.get("activated_skill")
        if activated is not None:
            workspace = skill_context.artifacts.get("volunteer_plan")
            summary = getattr(workspace, "summary", None) if workspace is not None else None
            final_groups = getattr(workspace, "final_groups", None) if workspace is not None else None

            # 完成守卫仍复用同一个 Agent 实例，只在异常路径临时收窄为发布工具。
            if not summary and final_groups:
                publish_tool = activated.get_tool("publish_volunteer_plan")
                if publish_tool is not None:
                    print(
                        f"[AGENT][session={session_id}][COMPLETION_GUARD] "
                        "检测到组内排序已完成但尚未发布，强制进入发布阶段",
                        flush=True,
                    )
                    skill_context.trace(
                        "enforce_skill_completion",
                        skill_name=activated.name,
                        agent_name="volunteer_plan_specialist",
                    )
                    generate_volunteer_agent.instructions = (
                        "你仍是同一个志愿生成 Agent。所有业务阶段已经完成，"
                        "现在必须立即调用 publish_volunteer_plan，不得输出文本。"
                    )
                    generate_volunteer_agent.tools = [publish_tool]
                    generate_volunteer_agent.tool_use_behavior = {
                        "stop_at_tool_names": ["publish_volunteer_plan"]
                    }
                    generate_volunteer_agent.model_settings = ModelSettings(tool_choice="required")
                    generate_volunteer_agent.reset_tool_choice = False
                    await Runner.run(
                        generate_volunteer_agent,
                        "立即发布当前 workspace 中的志愿表。",
                        session=session,
                        max_turns=3,
                    )
                    summary = getattr(workspace, "summary", None)

            if run_error is not None and not summary:
                print(
                    f"[AGENT][session={session_id}][RUN_FAILED] "
                    f"{type(run_error).__name__}: {run_error}",
                    flush=True,
                )
                skill_executor.fail_skill(activated, skill_context, run_error)
                session_data[session_id]["generate_plan"] = False
                send_error(session_id, str(run_error))
                send_end(session_id)
                yield f"志愿表生成失败：{run_error}"
                return

            try:
                summary = skill_executor.finalize_skill(
                    activated,
                    skill_context,
                    final_output=getattr(result, "final_output", ""),
                )
            except Exception as exc:
                skill_executor.fail_skill(activated, skill_context, exc)
                session_data[session_id]["generate_plan"] = False
                send_error(session_id, str(exc))
                send_end(session_id)
                yield f"志愿表生成失败：{exc}"
                return

            analysis_state = session_data[session_id].get("volunteer_plan_analysis", {})
            if analysis_state.get("status") != "success":
                print(
                    f"[AGENT][session={session_id}][ANALYSIS_START]",
                    flush=True,
                )
                await execute_volunteer_plan_analysis(skill_context)
                print(
                    f"[AGENT][session={session_id}][ANALYSIS_DONE]",
                    flush=True,
                )

            generate_volunteer_agent.instructions = base_generation_instructions
            generate_volunteer_agent.tools = [activate_skill]
            generate_volunteer_agent.tool_use_behavior = "run_llm_again"
            generate_volunteer_agent.model_settings = base_generation_model_settings
            generate_volunteer_agent.reset_tool_choice = True
            print(
                f"[AGENT][session={session_id}][REQUEST_DONE] summary={summary}",
                flush=True,
            )

            yield (
                "已为您生成志愿表并完成分析，"
                f"共选择 {summary.get('selected_groups', 0)} 个专业组、"
                f"{summary.get('selected_majors', 0)} 个专业。"
            )
        elif session_data[session_id].get("skill_activation_error"):
            yield f"Skill 激活失败：{session_data[session_id]['skill_activation_error']}"
        elif run_error is not None:
            raise run_error
    finally:
        await consulting_mcp_manager.cleanup_all()

async def run_terminal_chat(session_id: str, initial_message: str = "") -> None:
    """Run a stateful terminal conversation until the user enters exit."""
    print("\n=== 高考志愿填报 Agent 终端多轮对话 ===")
    print(f"会话 ID: {session_id}")
    print("输入 exit 结束对话。\n")

    pending_message = initial_message.strip()
    while True:
        try:
            user_query = pending_message or input("你: ").strip()
            pending_message = ""
        except (EOFError, KeyboardInterrupt):
            print("\n已退出 Agent。")
            return

        if not user_query:
            continue
        if user_query.lower() == "exit":
            print("已退出 Agent。")
            return

        response_chunks: list[str] = []
        try:
            async for chunk in multi_agent_chat(
                user_query,
                session_id,
                print_response_deltas=False,
            ):
                response_chunks.append(chunk)
        except Exception as exc:
            print(f"\nAgent 本轮执行失败: {type(exc).__name__}: {exc}\n", flush=True)
            continue

        response = "".join(response_chunks).strip()
        if response:
            print(f"\nAgent: {response}\n", flush=True)
        else:
            print("\nAgent: 本轮已执行完成。\n", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="高考志愿填报 Agent 终端多轮对话")
    parser.add_argument(
        "--session",
        default=get_current_session_id(),
        help="复用同一个 session ID 可延续对话记忆和 Skill workspace",
    )
    parser.add_argument(
        "message",
        nargs="*",
        help="可选的首轮消息；处理完成后仍会继续等待输入",
    )
    args = parser.parse_args()
    asyncio.run(run_terminal_chat(args.session, " ".join(args.message)))
