# README
`uv pip install -e .`

`uv sync --extra examples`


`ruff check .`

`pyright .`


## test

``` shell
uv run pytest tests/unit
uv run pytest tests/integration

uv run pytest tests/unit/examples_deep_agent_runtime -q

uv run pytest tests/integration/examples_deep_agent_runtime/test_web_tools_live.py -q
```

## trace
https://us.cloud.langfuse.com/project/cmmpuv8g0040dad07ihyem6vp/traces?peek=a998fe8525ebaa600a819b80a3c5cacd&timestamp=2026-03-14T06%3A04%3A12.008Z
