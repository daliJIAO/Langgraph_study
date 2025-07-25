# 🧮 LangGraph 智能表达式计算器

基于 LangGraph 和多Agent架构的智能数学表达式计算器，支持括号运算和详细的执行流程可视化。

## 📋 项目概述

本项目演示了如何使用 LangGraph 构建一个复杂的多Agent计算系统，将数学表达式的解析、路由和计算分离到不同的专用Agent中。系统能够：

- 🔍 **智能解析**：自动识别数学表达式中的运算优先级（括号优先）
- 🤖 **多Agent协作**：专用的加法和减法Agent分工合作
- 🗺️ **可视化流程**：详细展示每个节点的执行过程和Agent调用
- 🔄 **循环计算**：自动处理复杂表达式的逐步化简

## 🏗️ 系统架构

### 核心组件

```
📁 demo_work/
├── 🤖 agents/           # Agent模块
│   ├── base_agent.py    # 基础Agent类
│   ├── plus_agent.py    # 加法专用Agent
│   └── subtract_agent.py # 减法专用Agent
├── 🛠️ tools/           # 工具模块
│   ├── base_tool.py     # 工具注册和调用框架
│   ├── plus_tool.py     # 加法计算工具
│   └── subtract_tool.py # 减法计算工具
├── 🗺️ graphs/          # 图结构模块
│   ├── base_graph.py    # 基础图类（LangGraph包装器）
│   └── compute_graph.py # 计算图实现
├── 📊 states/           # 状态管理
│   └── multi_agent_state.py
└── 🧪 tests/           # 测试模块
```

### 计算图节点

- **ROUTER**: 表达式解析和路由分发中心
- **PLUS**: 普通加法运算节点（调用 PLUS_AGENT）
- **SUBTRACT**: 普通减法运算节点（调用 SUBTRACT_AGENT）
- **PLUS_BRACKET**: 括号内加法运算节点（调用 PLUS_AGENT）
- **SUBTRACT_BRACKET**: 括号内减法运算节点（调用 SUBTRACT_AGENT）

## 🚀 快速开始

### 环境准备

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **配置API密钥**
创建 `.env` 文件并添加：
```env
DASHSCOPE_API_KEY=your_api_key_here
```

### 运行示例

```bash
cd demo_work
python test_compute_graph.py
```

## 💡 使用示例

### 基础运算
```python
# 简单加法
3+5  # 输出: 8

# 简单减法
10-3  # 输出: 7
```

### 复杂表达式
```python
# 括号运算
(3+5)-2     # 输出: 6
((3+5)-2)+1 # 输出: 7

# 连续运算
10-3+2      # 输出: 9
```

## 🔍 执行流程演示

当您输入表达式 `(3+5)-2+1` 时，系统会显示详细的执行过程：

```
============================================================
🚀 节点执行: ROUTER (路由器)
📍 步骤 1: 分析表达式 '(3+5)-2+1'
============================================================
🔍 路由器分析: 括号内加法运算: 3 + 5
📍 路由器决策: 分发到 PLUS_BRACKET 节点
➡️  下一步: 执行 plus_bracket 节点

============================================================
🚀 节点执行: PLUS_BRACKET (括号内加法节点)
🧮 准备计算: 3 + 5
============================================================
📞 调用 Agent: PLUS_AGENT
📝 输入参数: "3+5"
🤖 Agent 原始输出: 8
💡 提取的计算结果: 8
✅ Agent 调用成功!
🔄 表达式更新: '(3+5)-2+1' → '8-2+1'
➡️  下一步: 返回 ROUTER 节点继续分析
```

## 🏛️ 技术架构

### Agent系统

#### BaseAgent
所有Agent的基础类，提供统一的接口和配置管理。

#### PlusAgent
```python
专用加法Agent，特点：
- 模型: qwen-plus
- 工具: plus_tool (精确到小数点后3位)
- 提示: 专门优化的加法计算提示词
```

#### SubtractAgent
```python
专用减法Agent，特点：
- 模型: qwen-plus  
- 工具: subtract_tool (精确到小数点后3位)
- 提示: 专门优化的减法计算提示词
```

### 工具系统

#### 工具注册机制
```python
@register_tool("plus", "计算两个浮点数的和", PlusSchema)
def plus_tool(a: float, b: float) -> float:
    return round(a + b, 3)
```

#### 统一调用接口
```python
# 调用工具并获取结构化结果
msg = invoke_tool("plus", {"a": 1.5, "b": 2.3})
# ToolMessage(tool='plus', success=True, result=3.8)
```

### 图结构系统

#### BaseGraph
LangGraph的轻量级包装器，特点：
- 🔧 **结构与功能分离**：先定义拓扑，后绑定函数
- 🎯 **类型安全**：完整的TypedDict状态管理
- 🔄 **灵活编译**：支持条件边和复杂路由

#### CalcState
```python
class CalcState(TypedDict, total=False):
    expr: str          # 当前表达式
    left: str          # 左操作数  
    right: str         # 右操作数
    route: str         # 路由决策
    span: tuple        # 操作位置
    result: str        # 最终结果
    op_type: str       # 操作类型（bracket/normal）
    step: int          # 步骤编号
    operation_log: str # 操作日志
```

## 🔄 执行流程详解

### 1. 表达式解析
```python
def parse_next_op(expr: str):
    # 优先处理括号内表达式
    bracket_match = re.search(r'\(([^()]+)\)', expr)
    if bracket_match:
        # 解析括号内的运算
        # 返回: left, op, right, span, "bracket"
    
    # 处理普通运算
    match = re.search(r'(-?\d+(?:\.\d+)?)\s*([+-])\s*(-?\d+(?:\.\d+)?)', expr)
    # 返回: left, op, right, span, "normal"
```

### 2. 路由决策
```python
def condition_router(state: CalcState) -> str:
    route = state.get("route", "end")
    # 返回: "plus", "subtract", "plus_bracket", "subtract_bracket", "end"
    return route
```

### 3. 表达式更新
```python
def _update_expression(state: CalcState, result: str) -> CalcState:
    # 根据op_type决定更新策略
    if state.get("op_type") == "bracket":
        # 替换整个括号
        new_expr = expr[:bracket_start] + result + expr[bracket_end:]
    else:
        # 普通替换
        new_expr = expr[:start] + result + expr[end:]
    return {"expr": new_expr}
```

## 🎯 特色功能

### 1. 实时流程追踪
- ✅ 节点执行顺序可视化
- ✅ Agent调用详情展示  
- ✅ 表达式变化追踪
- ✅ 错误处理和备用计算

### 2. 交互式计算器
```python
🎉 欢迎使用智能表达式计算器！
💡 支持括号运算，例如: (3+5)-2+1
💡 输入 'q' 退出

请输入数学表达式: (10-3)+2
# 显示详细执行过程...
# 最终结果: 9
```

### 3. 批量测试
```python
test_expressions = [
    "3+5",           # 简单加法
    "(3+5)-2",       # 括号优先
    "((3+5)-2)+1",   # 嵌套括号
    "10-3+2"         # 连续运算
]
```

## ���️ 扩展指南

### 添加新运算
1. **创建工具**
```python
@register_tool("multiply", "计算两数乘积", MultiplySchema)
def multiply_tool(a: float, b: float) -> float:
    return round(a * b, 3)
```

2. **创建Agent**
```python
class MultiplyAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            model="qwen-plus",
            tools=[multiply_tool],
            prompt="专门的乘法计算助手..."
        )
```

3. **扩展解析器**
```python
# 在 parse_next_op 中添加乘法匹配
op_match = re.search(r'(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)', expr)
```

4. **添加图节点**
```python
multiply_node = Node("multiply", func=run_multiply)
compute_graph.add_node(multiply_node)
```

### 自定义状态
```python
class CustomCalcState(CalcState):
    history: List[str]    # 计算历史
    precision: int        # 精度控制
    metadata: Dict        # 元数据
```

## 📊 性能特点

- ⚡ **高效解析**：正则表达式优化的O(n)解析
- 🔄 **循环稳定**：智能的表达式化简策略
- 🛡️ **错误容忍**：Agent失败时的备用计算
- 💾 **状态轻量**：最小化的状态传递

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [LangGraph](https://github.com/langchain-ai/langgraph) - 核心图执行引擎
- [Qwen](https://dashscope.console.aliyun.com/) - AI模型支持
- [Pydantic](https://pydantic-docs.helpmanual.io/) - 数据验证框架

---

📧 **联系方式**: 如有问题或建议，请创建 Issue 或 Pull Request。

⭐ **如果这个项目对您有帮助，请给个Star！**
