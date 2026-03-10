"""
DK-SAR 主编排逻辑
把三个智能体串联起来，实现"检索→生成→物理校验→修正"的完整闭环
对应论文 3.4 节：Multi-Agent Orchestration
"""
import json
from agents.context_agent import ContextAgent
from agents.rag_agent import RAGAgent
from agents.reflection_agent import ReflectionAgent
from utils.vector_store import get_vector_store
import config


def run_dk_sar(user_input: str, progress_callback=None) -> dict:
    """
    DK-SAR 主流程
    
    参数:
        user_input: 用户输入的化工异常场景描述
        progress_callback: 进度回调函数（可选），接受 (step, message) 参数
    
    返回: {
        "success": True/False,
        "report": HAZOP 分析报告 JSON,
        "context": 解析的工艺上下文,
        "reflection_history": 每轮反思记录,
        "error": 错误信息（失败时）
    }
    """
    def notify(step: str, message: str):
        print(f"[{step}] {message}")
        if progress_callback:
            progress_callback(step, message)
    
    notify("START", f"开始分析: {user_input[:50]}...")
    
    # ── 初始化智能体 ──────────────────────────────────────────
    context_agent = ContextAgent()
    rag_agent = RAGAgent()
    reflection_agent = ReflectionAgent()
    
    reflection_history = []
    
    try:
        # ── Step 1: 上下文解析 ─────────────────────────────────
        notify("CONTEXT", "智能体1：解析场景...")
        context_result = context_agent.run(user_input)
        context = context_result["context"]
        notify("CONTEXT_DONE", f"场景解析完成: {context['equipment']} / {context['parameter']} / {context['deviation_direction']}")
        
        # ── Step 2+3: RAG生成 + 物理反思循环 ─────────────────────
        report = None
        correction_guidance = ""
        
        for round_num in range(config.MAX_REFLECTION_ROUNDS):
            
            # RAG 生成报告
            notify("RAG", f"智能体2：RAG增强生成（第{round_num+1}轮）...")
            report = rag_agent.run(
                context=context,
                correction_guidance=correction_guidance,
                reflection_rounds=round_num
            )
            notify("RAG_DONE", f"报告初稿生成完成")
            
            # 物理反思校验
            notify("REFLECTION", f"智能体3：物理反思校验（第{round_num+1}轮）...")
            reflection_result = reflection_agent.run(report)
            
            reflection_history.append({
                "round": round_num + 1,
                "passed": reflection_result["passed"],
                "issues_count": len(reflection_result["issues"]),
                "issues": reflection_result["issues"]
            })
            
            if reflection_result["passed"]:
                notify("REFLECTION_DONE", f"✅ 物理校验通过（第{round_num+1}轮）")
                break
            else:
                issues_count = len(reflection_result["issues"])
                notify("REFLECTION_ISSUES", f"⚠️ 发现{issues_count}个物理问题，准备修正...")
                correction_guidance = reflection_result["correction_guidance"]
                
                # 更新报告的反思记录
                if "analysis_metadata" in report:
                    report["analysis_metadata"]["physical_issues_found"] = [
                        issue["description"] for issue in reflection_result["issues"]
                    ]
                
                if round_num == config.MAX_REFLECTION_ROUNDS - 1:
                    notify("MAX_ROUNDS", f"达到最大反思轮次({config.MAX_REFLECTION_ROUNDS})，使用最终版本")
        
        # 更新最终元数据
        if report and "analysis_metadata" in report:
            report["analysis_metadata"]["reflection_rounds"] = len(reflection_history)
            if reflection_history and reflection_history[-1]["passed"]:
                report["analysis_metadata"]["physical_issues_found"] = []
        
        notify("DONE", "✅ DK-SAR 分析完成！")
        
        return {
            "success": True,
            "report": report,
            "context": context,
            "reflection_history": reflection_history,
            "error": None
        }
        
    except Exception as e:
        error_msg = str(e)
        notify("ERROR", f"❌ 分析失败: {error_msg}")
        return {
            "success": False,
            "report": None,
            "context": None,
            "reflection_history": reflection_history,
            "error": error_msg
        }


def initialize_system():
    """
    系统初始化：构建向量索引
    首次运行必须调用，后续会自动跳过
    """
    print("\n" + "="*50)
    print("  DK-SAR 系统初始化")
    print("="*50)
    vs = get_vector_store()
    vs.build_index()
    print("="*50)
    print("  系统就绪 ✅")
    print("="*50 + "\n")


if __name__ == "__main__":
    # 直接运行此文件时，执行命令行测试
    initialize_system()
    
    test_input = "反应釜温度从80°C升至120°C，搅拌器仍在运行，冷却水流量正常"
    print(f"\n测试输入: {test_input}\n")
    
    result = run_dk_sar(test_input)
    
    if result["success"]:
        print("\n" + "="*50)
        print("分析报告:")
        print("="*50)
        print(json.dumps(result["report"], ensure_ascii=False, indent=2))
        print(f"\n反思历史: {result['reflection_history']}")
    else:
        print(f"\n❌ 失败: {result['error']}")
