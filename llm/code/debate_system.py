"""
基于LangGraph的多智能体辩论系统
辩论主题：AI是否应该在教育领域全面应用
包含：ReAct推理规划、短期记忆、长期记忆、工具调用
"""

import os
import sys
import json
from typing import TypedDict, List, Dict
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ollama
from langgraph.graph import StateGraph, END

# 导入工具和记忆
from tools.search_tool import search_debate_arguments
from tools.calculator import count_arguments, calculate_debate_score
from memory.memory import ShortTermMemory, LongTermMemory

# ==================== 配置 ====================
DEBATE_TOPIC = "AI是否应该在教育领域全面应用"
MODEL_NAME = "qwen3.5:2b"

# ==================== 初始化LLM ====================
print("正在初始化LLM...")
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """调用Ollama LLM"""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0.7}
    )
    return response["message"]["content"]

print("LLM初始化完成！\n")

# ==================== 定义状态 ====================
class DebateState(TypedDict):
    """辩论状态"""
    current_speaker: str
    debate_round: int
    affirmative_speech: str
    negative_speech: str
    debate_history: str
    final_judgment: str

# ==================== 初始化记忆 ====================
short_term_memory = ShortTermMemory(max_messages=50)
long_term_memory = LongTermMemory(storage_path="code/memory/debate_memory.json")

# ==================== Agent提示词 ====================
MODERATOR_SYSTEM_PROMPT = f"""你是一位专业的辩论赛主持人。辩论主题是：{DEBATE_TOPIC}

你的职责：
1. 开场介绍辩论主题和规则
2. 引导正反方交替发言
3. 维持辩论秩序
4. 最后邀请评委进行评判

请保持客观中立，语言专业简洁。"""

AFFIRMATIVE_SYSTEM_PROMPT = f"""你是辩论赛的正方辩手。你的立场是：AI应该在教育领域全面应用。

辩论主题：{DEBATE_TOPIC}

你的任务：
1. 提出支持AI在教育中全面应用的论点
2. 使用具体例子和数据支持你的观点
3. 使用ReAct方法：先思考(Thought)，再行动(Action-可以调用搜索工具获取论据)，观察结果(Observation)，最后给出发言(Final Answer)

请使用ReAct框架进行推理，你的发言应该有说服力，逻辑清晰，论据充分。"""

NEGATIVE_SYSTEM_PROMPT = f"""你是辩论赛的反方辩手。你的立场是：AI不应该在教育领域全面应用。

辩论主题：{DEBATE_TOPIC}

你的任务：
1. 提出反对AI在教育中全面应用的论点
2. 指出AI在教育中应用的局限性和风险
3. 反驳正方的观点
4. 提出更合理的替代方案

请保持理性客观，指出问题但不极端。你的发言应该有说服力，逻辑清晰。"""

JUDGE_SYSTEM_PROMPT = f"""你是一位经验丰富的辩论赛评委。辩论主题是：{DEBATE_TOPIC}

你的任务：
1. 仔细听取双方的论点
2. 评估双方的论证质量、逻辑性和说服力
3. 指出双方的优点和不足
4. 给出公正的评分和判决

评分标准：
- 论点质量（40%）
- 逻辑性（30%）
- 反驳能力（20%）
- 表达效果（10%）

请给出详细的评判理由和最终得分（正方得分/反方得分，满分100），并宣布获胜方。"""

# ==================== Agent节点函数 ====================
def moderator_node(state: DebateState) -> Dict:
    """主持人节点"""
    round_num = state.get("debate_round", 1)
    
    if round_num == 1:
        intro = f"""各位同学，大家好！

欢迎来到今天的辩论赛。我是本场辩论的主持人。

【辩论主题】{DEBATE_TOPIC}

【辩论规则】
1. 正方先发言，反方后发言
2. 每方有立论、反驳和总结三个环节
3. 最后由评委进行评判

现在，辩论正式开始！

首先，请正方一辩进行立论发言，阐述AI应该在教育领域全面应用的观点。"""
    else:
        intro = f"""感谢双方的精彩发言。

现在进入第{round_num}轮辩论。请双方继续进行辩论。"""
    
    short_term_memory.add_message("moderator", intro)
    print(f"\n{'='*60}")
    print(f"[主持人]")
    print(f"{'='*60}")
    print(intro)
    
    return {
        "current_speaker": "affirmative",
        "debate_round": round_num
    }

def affirmative_node(state: DebateState) -> Dict:
    """正方辩手节点 - 使用ReAct推理"""
    debate_history = state.get("debate_history", "")
    round_num = state.get("debate_round", 1)
    
    print(f"\n{'='*60}")
    print(f"[正方辩手 - 第{round_num}轮] (使用ReAct推理)")
    print(f"{'='*60}")
    print("正在思考中...")
    
    # 构建ReAct提示
    react_prompt = f"""当前是第{round_num}轮辩论。

之前的辩论历史：
{debate_history[-1000:] if len(debate_history) > 1000 else debate_history}

请使用ReAct框架进行发言：
Thought: 分析当前辩论情况
Action: （可选）决定是否需要搜索工具获取更多论据
Observation: （如果使用了工具）观察搜索结果
Final Answer: 你的正式发言内容"""
    
    speech = call_llm(AFFIRMATIVE_SYSTEM_PROMPT, react_prompt)
    print(speech)
    
    # 存储到记忆
    short_term_memory.add_message("affirmative", speech)
    long_term_memory.add_argument("affirmative", speech)
    
    # 更新辩论历史
    new_history = debate_history + f"\n【正方发言-第{round_num}轮】\n{speech}\n"
    
    return {
        "affirmative_speech": speech,
        "debate_history": new_history,
        "current_speaker": "negative"
    }

def negative_node(state: DebateState) -> Dict:
    """反方辩手节点"""
    debate_history = state.get("debate_history", "")
    round_num = state.get("debate_round", 1)
    affirmative_speech = state.get("affirmative_speech", "")
    
    print(f"\n{'='*60}")
    print(f"[反方辩手 - 第{round_num}轮]")
    print(f"{'='*60}")
    print("正在思考中...")
    
    prompt = f"""当前是第{round_num}轮辩论。

正方刚才的发言：
{affirmative_speech[-800:] if len(affirmative_speech) > 800 else affirmative_speech}

请针对正方的观点进行反驳，并阐述你方的立场。"""
    
    speech = call_llm(NEGATIVE_SYSTEM_PROMPT, prompt)
    print(speech)
    
    # 存储到记忆
    short_term_memory.add_message("negative", speech)
    long_term_memory.add_argument("negative", speech)
    
    # 更新辩论历史
    new_history = debate_history + f"\n【反方发言-第{round_num}轮】\n{speech}\n"
    
    return {
        "negative_speech": speech,
        "debate_history": new_history
    }

def judge_node(state: DebateState) -> Dict:
    """评委节点"""
    debate_history = state.get("debate_history", "")
    
    print(f"\n{'='*60}")
    print(f"[评委评判]")
    print(f"{'='*60}")
    print("正在评判中...")
    
    prompt = f"""请根据以下完整的辩论记录进行评判：

{debate_history}

请按照以下格式给出评判：
1. 正方优点分析
2. 反方优点分析
3. 正方不足分析
4. 反方不足分析
5. 最终评分（正方得分/反方得分，满分100）
6. 获胜方
7. 评判总结"""
    
    judgment = call_llm(JUDGE_SYSTEM_PROMPT, prompt)
    print(judgment)
    
    # 存储结论
    long_term_memory.add_conclusion(judgment)
    
    return {
        "final_judgment": judgment
    }

# ==================== 条件路由 ====================
def route_after_negative(state: DebateState) -> str:
    """反方发言后的路由"""
    round_num = state.get("debate_round", 1)
    print(f"\n>>> 当前轮次: {round_num}")
    if round_num >= 3:
        print(">>> 辩论结束，进入评判阶段")
        return "judge"
    else:
        print(f">>> 继续第{round_num + 1}轮辩论")
        return "continue"

# ==================== 构建图 ====================
def create_debate_graph():
    """创建辩论工作流图"""
    workflow = StateGraph(DebateState)
    
    # 添加节点
    workflow.add_node("moderator", moderator_node)
    workflow.add_node("affirmative", affirmative_node)
    workflow.add_node("negative", negative_node)
    workflow.add_node("judge", judge_node)
    
    # 设置入口点
    workflow.set_entry_point("moderator")
    
    # 添加边
    workflow.add_edge("moderator", "affirmative")
    workflow.add_edge("affirmative", "negative")
    
    # 条件边：反方发言后决定是继续还是结束
    workflow.add_conditional_edges(
        "negative",
        route_after_negative,
        {
            "continue": "moderator",
            "judge": "judge"
        }
    )
    
    workflow.add_edge("judge", END)
    
    return workflow.compile()

# ==================== 主函数 ====================
def run_debate():
    """运行辩论系统"""
    print("=" * 60)
    print("基于LangGraph的多智能体辩论系统")
    print(f"辩论主题：{DEBATE_TOPIC}")
    print("=" * 60)
    print("\n正在初始化辩论系统...\n")
    
    # 创建图
    app = create_debate_graph()
    
    # 初始状态
    initial_state = {
        "current_speaker": "moderator",
        "debate_round": 1,
        "affirmative_speech": "",
        "negative_speech": "",
        "debate_history": "",
        "final_judgment": ""
    }
    
    print("\n辩论开始！\n")
    
    # 运行辩论
    final_state = None
    round_counter = 1
    
    for event in app.stream(initial_state):
        for node_name, output in event.items():
            # 更新轮次
            if node_name == "negative":
                round_counter += 1
                output["debate_round"] = round_counter
            
            final_state = output
    
    print("\n\n" + "=" * 60)
    print("辩论结束！")
    print("=" * 60)
    
    # 保存结果
    if final_state:
        result = {
            "topic": DEBATE_TOPIC,
            "debate_history": final_state.get("debate_history", ""),
            "final_judgment": final_state.get("final_judgment", ""),
            "timestamp": datetime.now().isoformat()
        }
        
        os.makedirs("code/output", exist_ok=True)
        with open("code/output/debate_result.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print("\n辩论结果已保存到 code/output/debate_result.json")
        print("记忆文件已保存到 code/memory/debate_memory.json")
    
    return final_state

if __name__ == "__main__":
    try:
        run_debate()
    except Exception as e:
        print(f"\n运行出错: {e}")
        import traceback
        traceback.print_exc()
