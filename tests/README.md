# 测试目录

## 自动化回归测试

`tests/` 根目录下的 `test_*.py` 不调用真实志愿 API，可统一运行：

```powershell
python -X utf8 -m unittest discover -s tests -v
```

## 真实环境冒烟测试

`tests/live/` 中的脚本用于手动验证真实模型、MCP、HTTP/SSE 或志愿 API，不会被上面的自动化命令发现。

```powershell
python -X utf8 tests/live/agent_smoke.py
python -X utf8 tests/live/consulting_mcp_smoke.py
python -X utf8 tests/live/skill_http_smoke.py
python -X utf8 tests/live/skill_resume_smoke.py
python -X utf8 tests/live/form_api_smoke.py
```

后三个 HTTP 脚本需要先启动 `app.py`；`agent_smoke.py` 和 `consulting_mcp_smoke.py` 不需要 Flask。
