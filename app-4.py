"""
DK-SAR 网页界面
使用 Gradio 构建，运行后在浏览器打开 http://127.0.0.1:7860
"""
import os
# 解决国内网络问题（必须在所有其他 import 之前）
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import gradio as gr
from main import run_dk_sar, initialize_system

# ── 预设示例场景 ───────────────────────────────────────────────────────────────
EXAMPLES = [
    "反应釜温度从80°C升至120°C，搅拌器仍在运行，冷却水流量正常",
    "换热器出口温度持续升高，超过设计温度15°C，操作人员停止了搅拌",
    "储罐液位计显示液位正常，但进料流量持续大于出料流量，差值约20%",
    "管道发生少量可燃气体泄漏，现场人员认为气体会自然扩散，未采取措施",
    "反应器压力上升，泄压阀自动开启，操作人员判断系统已经安全继续操作",
]

# ── 报告渲染函数 ───────────────────────────────────────────────────────────────
def render_report(report: dict, context: dict, reflection_history: list) -> str:
    """将 JSON 报告渲染为 Markdown 格式"""
    if not report:
        return "❌ 报告生成失败"
    
    lines = []
    
    # 场景摘要
    node = report.get("node_info", {})
    lines.append(f"## 📋 HAZOP 分析报告")
    lines.append(f"\n**设备**: {node.get('equipment', 'N/A')}  |  "
                 f"**参数**: {node.get('parameter', 'N/A')}  |  "
                 f"**偏差方向**: {node.get('deviation_direction', 'N/A')}")
    lines.append(f"\n**正常值**: {node.get('normal_value', 'N/A')}  |  "
                 f"**当前值**: {node.get('current_value', 'N/A')}")
    
    lines.append("\n---")
    
    # 分析内容
    for deviation in report.get("deviations", []):
        
        # 原因
        lines.append("\n### 🔍 可能原因")
        for i, cause in enumerate(deviation.get("causes", []), 1):
            lines.append(f"{i}. **[{cause.get('type', '')}]** {cause.get('description', '')}")
        
        # 后果
        lines.append("\n### ⚠️ 后果链")
        consequences = deviation.get("consequences", [])
        for i, c in enumerate(consequences):
            arrow = "→" if i > 0 else "①"
            lines.append(f"{arrow} {c}")
        
        # 保护措施
        lines.append("\n### 🛡️ 现有保护措施")
        for sg in deviation.get("safeguards", []):
            effectiveness = sg.get('effectiveness', '')
            icon = "✅" if effectiveness == "有效" else ("⚠️" if effectiveness == "部分有效" else "❌")
            lines.append(f"- {icon} {sg.get('measure', '')} （{effectiveness}）")
        
        # 建议措施
        lines.append("\n### 💡 建议措施")
        for rec in deviation.get("recommendations", []):
            priority = rec.get('priority', '')
            priority_icon = "🔴" if priority == "高" else ("🟡" if priority == "中" else "🟢")
            lines.append(f"- {priority_icon} **[{priority}]** {rec.get('action', '')}")
    
    lines.append("\n---")
    
    # 元数据
    meta = report.get("analysis_metadata", {})
    lines.append("\n### 📊 分析元数据")
    
    ref_cases = meta.get("referenced_cases", [])
    if ref_cases:
        lines.append(f"- **参考案例**: {', '.join(ref_cases)}")
    
    rounds = meta.get("reflection_rounds", 0)
    lines.append(f"- **物理反思轮次**: {rounds} 轮")
    
    confidence = meta.get("confidence_level", "N/A")
    lines.append(f"- **分析置信度**: {confidence}")
    
    issues = meta.get("physical_issues_found", [])
    if issues:
        lines.append(f"- **已修正的物理问题**: {len(issues)} 个")
    else:
        lines.append(f"- **物理校验**: ✅ 通过，无重大问题")
    
    return "\n".join(lines)


def render_process_log(context: dict, reflection_history: list) -> str:
    """渲染执行过程日志"""
    lines = []
    
    if context:
        lines.append("### 🔍 场景解析结果")
        lines.append(f"```json\n{json.dumps(context, ensure_ascii=False, indent=2)}\n```")
    
    if reflection_history:
        lines.append("\n### 🔬 物理反思记录")
        for h in reflection_history:
            round_num = h["round"]
            passed = h["passed"]
            issues_count = h["issues_count"]
            
            status = "✅ 通过" if passed else f"⚠️ 发现 {issues_count} 个问题"
            lines.append(f"\n**第 {round_num} 轮**: {status}")
            
            if h.get("issues"):
                for issue in h["issues"][:3]:  # 最多显示3个
                    lines.append(f"  - [{issue.get('issue_type')}] {issue.get('description', '')[:80]}...")
    
    return "\n".join(lines) if lines else "暂无执行记录"


# ── 主分析函数 ─────────────────────────────────────────────────────────────────
def analyze(user_input: str) -> tuple:
    """Gradio 调用的主函数"""
    if not user_input or not user_input.strip():
        return "⚠️ 请输入化工异常场景描述", "请在左侧输入框输入内容", ""
    
    # 执行分析
    result = run_dk_sar(user_input.strip())
    
    if result["success"]:
        report_md = render_report(
            result["report"],
            result["context"],
            result["reflection_history"]
        )
        process_md = render_process_log(
            result["context"],
            result["reflection_history"]
        )
        raw_json = json.dumps(result["report"], ensure_ascii=False, indent=2)
        return report_md, process_md, raw_json
    else:
        error_msg = f"## ❌ 分析失败\n\n**错误信息**: {result['error']}\n\n**可能原因**:\n- API Key 未配置（检查 .env 文件）\n- 网络连接问题\n- 模型调用超时"
        return error_msg, "", ""


# ── Gradio 界面定义 ────────────────────────────────────────────────────────────
def create_ui():
    with gr.Blocks(title="DK-SAR 化工安全分析系统") as demo:
        
        # 标题
        gr.HTML("""
        <div class="title-bar">
            <h1>🏭 DK-SAR 化工安全分析系统</h1>
            <p>双知识增强自适应推理 · 自动化 HAZOP 分析 · 多智能体驱动</p>
            <p style="color: #888; font-size: 12px;">
                基于论文：DK-SAR: Dual Knowledge-enhanced Self-Adaptive Reasoning for Automated HAZOP Analysis
            </p>
        </div>
        """)
        
        with gr.Row():
            # 左侧：输入区
            with gr.Column(scale=1):
                gr.Markdown("### 📝 输入化工异常场景")
                
                user_input = gr.Textbox(
                    label="场景描述",
                    placeholder="例如：反应釜温度从80°C升至120°C，搅拌器仍在运行...",
                    lines=5
                )
                
                analyze_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")
                
                gr.Markdown("**💡 点击加载示例场景：**")
                for i, example in enumerate(EXAMPLES):
                    btn = gr.Button(
                        f"示例{i+1}: {example[:40]}...",
                        size="sm",
                        variant="secondary"
                    )
                    btn.click(fn=lambda e=example: e, outputs=user_input)
            
            # 右侧：输出区
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📋 分析报告"):
                        report_output = gr.Markdown(label="HAZOP 分析报告")
                    
                    with gr.Tab("🔬 执行过程"):
                        process_output = gr.Markdown(label="执行日志")
                    
                    with gr.Tab("📄 原始 JSON"):
                        json_output = gr.Code(
                            label="完整 JSON 报告",
                            language="json"
                        )
        
        # 绑定事件
        analyze_btn.click(
            fn=analyze,
            inputs=[user_input],
            outputs=[report_output, process_output, json_output]
        )
        
        # 底部说明
        gr.Markdown("""
        ---
        **系统架构**: 上下文智能体 → RAG增强分析智能体 ⇄ 物理反思智能体（最多3轮修正）
        
        **数据来源**: 正样本库（CSB事故案例）| 负样本库（物理谬误库）| 模式仓库（IEC 61882 Schema）
        
        **GitHub**: [github.com/your-username/dk-sar](https://github.com/your-username/dk-sar)
        """)
    
    return demo


# ── 主程序入口 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DK-SAR 化工安全分析系统 启动中...")
    print("="*60)
    
    # 初始化（构建向量索引）
    initialize_system()
    
    # 启动网页
    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",  # 只监听本机，避免代理问题
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),   # Gradio 6.x: theme 移到 launch()
    )
