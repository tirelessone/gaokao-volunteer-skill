from flask import Flask, Response, request, jsonify
import json
import queue
from collections import deque
import asyncio
from event_queue import get_or_create_queue, remove_queue
from sqlalchemy.ext.asyncio import create_async_engine
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession
from skill_runtime import get_skill_registry
from volunteer_planner.service import generate_plan


engine = ""
# engine = create_async_engine("mysql+aiomysql://ssb_nextgoo:jyJEPnzmakcwk8kr@rm-bp15wkjnnu9a03550.mysql.rds.aliyuncs.com:3306/agents")
engine = create_async_engine("mysql+aiomysql://root:12345@localhost:3306/agents")
app = Flask(__name__)


def event_generator(session_id: str):
    q = get_or_create_queue(session_id)
    try:
        print(f"[DEBUG] 开始为 session {session_id} 生成事件流")
        yield ": connected\n\n"
        while True:
            try:
                try:
                    event = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                print(f"[DEBUG] 发送事件: {event.get('type')}")
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") == "end":
                    return
            except Exception as e:
                print(f"[DEBUG] 发生在while循环中，处理事件时出错: {e}")
                continue
                
    except GeneratorExit:
        print(f"[DEBUG] 事件流被取消: {session_id}")
    finally:
        remove_queue(session_id)
        print(f"[DEBUG] 清理 session {session_id} 的队列")

# === 单一 SSE 接口 ===
@app.route('/api/stream')
def stream():
    session_id = request.args.get('session_id')
    if not session_id:
        return "Missing session_id", 400

    # 先移除旧队列，确保干净
    remove_queue(session_id)
    get_or_create_queue(session_id)

    return Response(
        event_generator(session_id),
        mimetype='text/event-stream',
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.route('/api/skills', methods=['GET'])
def list_project_skills():
    """Expose lightweight project Skill metadata without loading full instructions."""
    registry = get_skill_registry()
    return jsonify({
        "status": "success",
        "skills": [item.to_dict() for item in registry.list_metadata()]
    })


# === Chat 接口 ===
@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    try:
        from multi_agent import multi_agent_chat
        data = request.json
        user_input = data.get("message", "")
        session_id = data.get("session_id", "default_session")
        print(f"传给multi_agent_chat的session_id为：{session_id}\n")

        def chat_generator():
            import asyncio
            from collections import deque
            
            # 创建事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # 用于存储生成的数据
            chunks_queue = deque()
            exception = None
            completed = False
            
            async def consume_async_generator():
                nonlocal exception, completed
                try:
                    full_response = ""
                    async for chunk in multi_agent_chat(user_input, session_id):
                        full_response += chunk
                        
                        # 动态读取 generate_plan 标记
                        try:
                            from multi_agent import session_data as _session_data
                            generate_plan = bool(_session_data.get(session_id, {}).get("generate_plan", False))
                        except Exception:
                            generate_plan = False

                        chunk_data = {
                            "type": "content",
                            "content": chunk,
                            "session_id": session_id,
                            "generate_plan": generate_plan
                        }
                        chunks_queue.append(('chunk', chunk_data))
                    
                    # 添加结束事件
                    final_data = {
                        "type": "end",
                        "session_id": session_id,
                        "full_response": full_response,
                        "generate_plan": generate_plan
                    }
                    chunks_queue.append(('end', final_data))
                    
                except Exception as e:
                    exception = e
                    error_data = {
                        "type": "error",
                        "error": str(e),
                        "session_id": session_id
                    }
                    chunks_queue.append(('error', error_data))
                finally:
                    completed = True
            
            # 启动异步任务
            task = loop.create_task(consume_async_generator())
            
            try:
                # 同步生成器部分
                while not completed or chunks_queue:
                    while chunks_queue:
                        item_type, data = chunks_queue.popleft()
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                        if item_type in ['end', 'error']:
                            return
                    
                    # 短暂等待新数据
                    loop.run_until_complete(asyncio.sleep(0.01))
                    
            except GeneratorExit:
                # 客户端断开连接，取消任务
                if not task.done():
                    task.cancel()
                raise
            except Exception as e:
                error_data = {
                    "type": "error",
                    "error": str(e),
                    "session_id": session_id
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            finally:
                try:
                    if not task.done():
                        task.cancel()
                    loop.run_until_complete(asyncio.sleep(0.1))
                    loop.close()
                except:
                    pass

        return Response(
            chat_generator(),
            mimetype='text/event-stream',
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive"
            }
        )

    except Exception as e:
        error_response = {"error": str(e)}
        return Response(
            json.dumps(error_response, ensure_ascii=False),
            status=500,
            content_type='application/json; charset=utf-8'
        )

# === 清空记忆接口 ===
@app.route("/api/clear_memory", methods=["POST"])
def clear_memory():
    try:
        data = request.json
        session_id = data.get("session_id")
        #测试环境：
        #engine = create_async_engine("mysql+aiomysql://ssb_nextgoo:jyJEPnzmakcwk8kr@localhost:3306/agents")
        #engine = create_async_engine("mysql+aiomysql://ssb_nextgoo:jyJEPnzmakcwk8kr@rm-bp15wkjnnu9a03550.mysql.rds.aliyuncs.com:3306/agents")
        engine = create_async_engine("mysql+aiomysql://root:12345@localhost:3306/agents")
        # 先清空数据库里的会话记忆
        session = SQLAlchemySession(
            session_id,
            engine=engine,
            create_tables=False,  # 你手动建表了，所以保持 False
        )

        asyncio.run(session.clear_session())

        # 再清除内存里的队列
        remove_queue(session_id)
        from multi_agent import session_data
        print("已导入全局变量session_data")
        if session_id in session_data:
            del session_data[session_id]
            # print("内存记忆已清除")

        print(f"✅ 清空记忆成功: {session_id}")
        return jsonify({"status": "success", "message": f"记忆和队列已清空: {session_id}"})

    except Exception as e:
        print(f"❌ 清空记忆失败: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

async def add_memory_to_session(memory, session_id):
    try:
        # 构建符合格式的记忆消息（role 为 assistant 表示智能体的记忆）
        memory_item = {
            "role": "assistant",
            "content": memory
        }

        # 初始化 MySQL 引擎
        # engine = create_async_engine("mysql+aiomysql://ssb_nextgoo:jyJEPnzmakcwk8kr@rm-bp15wkjnnu9a03550.mysql.rds.aliyuncs.com:3306/agents")
        engine = create_async_engine("mysql+aiomysql://root:12345@localhost:3306/agents")

        # 初始化 SQLAlchemySession
        session = SQLAlchemySession(
            session_id=session_id,
            engine=engine,
            create_tables=False,
        )

        # 存入记忆（在异步函数中使用 await 合法）
        await session.add_items([memory_item])
        print("记忆添加成功！")

        # 关闭引擎（释放资源）
        await engine.dispose()

    except Exception as e:
        print(f"添加记忆失败：{str(e)}")


# === 表格数据生成志愿表接口 ===
@app.route('/api/generate_plan_from_form', methods=['POST'])
def generate_plan_from_form():
    """
    接收前端表格数据，直接生成志愿表
    
    请求体示例：
    {
        "session_id": "test",
        "major_list": "数学类",
        "firstChoice": "物",
        "score": 650,
        "recruit": 0,
        "province": "四川省",
        "foreign": 0,
        "reselection": "化、生、不限",
        "matriculation": 0,
        "target": 0,
        "healthCheckupLimit": "000",
        "notMajorSecondDesc": "法学类",
        "universitySecondDesc": "河南大学,海南大学",
        "citySecondDesc": "成都市,北京市",
        "notCitySecondDesc": "绵阳市,广元市"
        "dimension":0
        "nature":0"
    }
    """
    try:
        # 获取POST请求体中的JSON数据
        data = request.json or {}
        
        session_id = data.get('session_id', 'default_session')
        major_list = data.get('major_list', '')
        firstChoice = data.get('firstChoice', '物')
        score = data.get('score')
        recruit = data.get('recruit', 0)
        province = data.get('province', '四川省')
        foreign = data.get('foreign', 0)
        reselection = data.get('reselection', '')
        matriculation = data.get('matriculation', 1)
        target = data.get('target', 1)
        healthCheckupLimit= data.get('healthCheckupLimit', '000')
        notMajorSecondDesc = data.get('notMajorSecondDesc', '')
        universitySecondDesc = data.get('universitySecondDesc', '')
        citySecondDesc = data.get('citySecondDesc', '')
        notCitySecondDesc = data.get('notCitySecondDesc', '')
        dimension = data.get('dimension', 0)
        nature = data.get('nature', None)
        

        # 参数验证
        if not score:
            return jsonify({
                "status": "error",
                "message": "缺少必需参数：score（分数）"
            }), 400
        
        if not major_list:
            return jsonify({
                "status": "error",
                "message": "缺少必需参数：major_list（意向专业）"
            }), 400
        
        # 处理专业列表（可能是字符串、数组或逗号分隔的字符串）
        if isinstance(major_list, str):
            major_list_array = [m.strip() for m in major_list.split(',') if m.strip()]
        elif isinstance(major_list, list):
            major_list_array = major_list
        else:
            major_list_array = [str(major_list)]
        
        # 处理副科信息（可能是字符串、数组或顿号分隔的字符串）
        if isinstance(reselection, str):
            reselection_array = [r.strip() for r in reselection.split('、') if r.strip()]
        elif isinstance(reselection, list):
            reselection_array = reselection
        else:
            reselection_array = ['不限']

        # 确保reselection包含"不限"并且在首位
        if '不限' not in reselection_array:
            reselection_array.insert(0, '不限')
        elif reselection_array[0] != '不限':
            reselection_array.remove('不限')
            reselection_array.insert(0, '不限')

        if dimension == 2:
            if major_list_array:
                dimension = 0
            else:
                dimension = 1

        # 构造格式化的用户信息字典
        formatted_user_data = {
            "major": major_list_array,
            "score": score,
            "firstChoice": firstChoice,
            "reselection": reselection_array,
            "foreign": foreign,
            "recruit": recruit,
            "province": province,
            "matriculation": matriculation,
            "target": target,
            "city": citySecondDesc,
            "dimension": dimension,
            "healthCheckupLimit": healthCheckupLimit,
            "not_major": notMajorSecondDesc,
            "university": universitySecondDesc,
            "not_city": notCitySecondDesc,
            "nature": nature
        }
        
        print(f"🔄 [DEBUG] 转换后的格式化数据:")
        print(f"  {formatted_user_data}")
        
        print(f"🚀 [DEBUG] 开始生成志愿表... (session_id: {session_id})")
        plan_result = generate_plan(
            formatted_user_data,
            session_id,
            plan_policy=data.get('plan_policy'),
            emit_events=True,
        )
        true_major_group_count = plan_result.statistics.selected_groups
        if true_major_group_count <= 20:
            preference_label = "专业" if dimension == 0 else "城市或院校"
            full_response = (
                f"当前生成了{true_major_group_count}个专业组。"
                f"建议适当放宽{preference_label}、地区、院校性质或合作办学限制后重新生成。"
            )
        else:
            full_response = f"已生成{true_major_group_count}个专业组，并按您的策略完成排序和筛选。"

        if plan_result.status == "success":
            print(f"✅ [DEBUG] 志愿表生成成功，长度: {len(plan_result.plan_text)} 字符")
            print(f"✅ [DEBUG] 存入记忆")
            memory = (
                f"用户信息：{json.dumps(formatted_user_data, ensure_ascii=False)}\n"
                f"志愿表摘要：{json.dumps(plan_result.model_summary(), ensure_ascii=False)}"
            )
            asyncio.run(add_memory_to_session(memory, session_id))
            return jsonify({
                "status": "success",
                "session_id": session_id,
                "full_response": full_response,
                "metadata": {
                    "policy": plan_result.policy.model_dump(),
                    "statistics": plan_result.statistics.model_dump()
                }
            })
        else:
            print(f"❌ [DEBUG] 志愿表生成失败")
            return jsonify({
                "status": "error",
                "message": "志愿表生成失败，请检查输入参数"
            }), 500
    
    except Exception as e:
        print(f"❌ [DEBUG] 接口调用出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"服务器错误: {str(e)}"
        }), 500

        
# === 启动入口 ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
