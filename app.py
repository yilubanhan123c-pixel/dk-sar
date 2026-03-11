"""
DK-SAR 网页界面 v2
- 实时进度流（每步执行状态即时显示）
- 报告导出（下载 JSON）
- 论文指标评估面板（PCC / CCC / RDI / LCC）
"""
import os
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import tempfile
import gradio as gr

from main import initialize_system
from agents.context_agent import ContextAgent
from agents.rag_agent import RAGAgent
from agents.reflection_agent import ReflectionAgent
from langgraph.graph import StateGraph, END
from eval import evaluate_report, format_scores_markdown
import config

# ── 预设示例场景 ──────────────────────────────────────────────
EXAMPLES = [
    "反应釜温度从80°C升至120°C，搅拌器仍在运行，冷却水流量正常",
    "换热器出口温度持续升高，超过设计温度15°C，操作人员停止了搅拌",
    "储罐液位计显示液位正常，但进料流量持续大于出料流量，差值约20%",
    "管道发生少量可燃气体泄漏，现场人员认为气体会自然扩散，未采取措施",
    "反应器压力上升，泄压阀自动开启，操作人员判断系统已经安全继续操作",
]


# ════════════════════════════════════════════════════════════════
#  报告渲染
# ════════════════════════════════════════════════════════════════

def render_report(report: dict) -> str:
    if not report:
        return "❌ 报告生成失败"
    lines = []
    node = report.get("node_info", {})
    lines.append("## 📋 HAZOP 分析报告")
    lines.append(
        f"\n**设备**: {node.get('equipment','N/A')}  |  "
        f"**参数**: {node.get('parameter','N/A')}  |  "
        f"**偏差方向**: {node.get('deviation_direction','N/A')}"
    )
    lines.append(
        f"\n**正常值**: {node.get('normal_value','N/A')}  |  "
        f"**当前值**: {node.get('current_value','N/A')}"
    )
    lines.append("\n---")
    for deviation in report.get("deviations", []):
        lines.append("\n### 🔍 可能原因")
        for i, c in enumerate(deviation.get("causes", []), 1):
            lines.append(f"{i}. **[{c.get('type','')}]** {c.get('description','')}")
        lines.append("\n### ⚠️ 后果链")
        for i, c in enumerate(deviation.get("consequences", [])):
            lines.append(f"{'①' if i==0 else '→'} {c}")
        lines.append("\n### 🛡️ 现有保护措施")
        for sg in deviation.get("safeguards", []):
            eff = sg.get('effectiveness','')
            icon = "✅" if eff=="有效" else ("⚠️" if eff=="部分有效" else "❌")
            lines.append(f"- {icon} {sg.get('measure','')} （{eff}）")
        lines.append("\n### 💡 建议措施")
        for rec in deviation.get("recommendations", []):
            p = rec.get('priority','')
            icon = "🔴" if p=="高" else ("🟡" if p=="中" else "🟢")
            lines.append(f"- {icon} **[{p}]** {rec.get('action','')}")
    lines.append("\n---")
    meta = report.get("analysis_metadata", {})
    lines.append("\n### 📊 分析元数据")
    ref = meta.get("referenced_cases", [])
    if ref:
        lines.append(f"- **参考案例**: {', '.join(ref)}")
    lines.append(f"- **物理反思轮次**: {meta.get('reflection_rounds', 0)} 轮")
    lines.append(f"- **分析置信度**: {meta.get('confidence_level','N/A')}")
    issues = meta.get("physical_issues_found", [])
    if issues:
        lines.append(f"- **已修正物理问题**: {len(issues)} 个")
    else:
        lines.append("- **物理校验**: ✅ 通过，无重大问题")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
#  核心：带实时进度的分析函数（generator，用于 Gradio streaming）
# ════════════════════════════════════════════════════════════════

def analyze_streaming(user_input: str):
    """
    用 Python generator + Gradio streaming 实现实时进度显示
    每执行完一个智能体就 yield 一次，前端即时更新
    """
    if not user_input or not user_input.strip():
        yield "⚠️ 请输入场景描述", "", "", "", None
        return

    progress_log = []
    report = None
    context = None
    reflection_history = []

    def log(msg: str):
        progress_log.append(msg)
        return "\n".join(progress_log)

    # ── 智能体1 ──────────────────────────────────────────────
    progress = log("⏳ **[节点1 / 3]** 上下文智能体 — 解析场景...")
    yield "分析中，请稍候...", progress, "", "", None

    try:
        ctx_agent = ContextAgent()
        ctx_result = ctx_agent.run(user_input.strip())
        context = ctx_result["context"]
        progress = log(
            f"✅ **节点1 完成** — {context['equipment']} / "
            f"{context['parameter']} / {context['deviation_direction']}"
        )
    except Exception as e:
        progress = log(f"❌ 节点1 失败: {e}")
        yield "❌ 分析失败", progress, "", "", None
        return

    yield "分析中，请稍候...", progress, "", "", None

    # ── 智能体2+3 循环 ─────────────────────────────────────────
    correction_guidance = ""
    round_num = 0

    while round_num < config.MAX_REFLECTION_ROUNDS:
        # 节点2
        progress = log(f"\n⏳ **[节点2 / 3]** RAG分析智能体 — 第 {round_num+1} 轮生成...")
        yield "分析中，请稍候...", progress, "", "", None

        try:
            rag_agent = RAGAgent()
            report = rag_agent.run(
                context=context,
                correction_guidance=correction_guidance,
                reflection_rounds=round_num,
            )
            ref_cases = report.get("analysis_metadata", {}).get("referenced_cases", [])
            progress = log(f"✅ **节点2 完成** — 参考案例: {ref_cases}")
        except Exception as e:
            progress = log(f"❌ 节点2 失败: {e}")
            yield "❌ 分析失败", progress, "", "", None
            return

        yield "分析中，请稍候...", progress, "", "", None

        # 节点3
        progress = log(f"\n⏳ **[节点3 / 3]** 物理反思智能体 — 第 {round_num+1} 轮校验...")
        yield "分析中，请稍候...", progress, "", "", None

        try:
            ref_agent = ReflectionAgent()
            ref_result = ref_agent.run(report)
            reflection_history.append({
                "round": round_num + 1,
                "passed": ref_result["passed"],
                "issues_count": len(ref_result["issues"]),
                "issues": ref_result["issues"],
            })

            if ref_result["passed"]:
                progress = log("✅ **节点3 通过** — 物理校验无问题，分析完成！")
                yield "分析中，请稍候...", progress, "", "", None
                break
            else:
                n = len(ref_result["issues"])
                progress = log(
                    f"⚠️ **节点3 发现 {n} 个物理问题** — 返回节点2修正\n"
                    + "\n".join(f"  - {i['description'][:60]}..." for i in ref_result["issues"][:2])
                )
                correction_guidance = ref_result["correction_guidance"]
                round_num += 1

                if round_num >= config.MAX_REFLECTION_ROUNDS:
                    progress = log(f"⚠️ 已达最大轮次 {config.MAX_REFLECTION_ROUNDS}，使用当前版本")
                yield "分析中，请稍候...", progress, "", "", None

        except Exception as e:
            progress = log(f"❌ 节点3 失败: {e}")
            break

    # ── 生成最终输出 ──────────────────────────────────────────
    if not report:
        yield "❌ 分析失败，未生成报告", progress, "", "", None
        return

    # 更新元数据
    if "analysis_metadata" not in report:
        report["analysis_metadata"] = {}
    report["analysis_metadata"]["reflection_rounds"] = round_num + 1

    # 渲染报告
    report_md = render_report(report)

    # 评估分数
    try:
        scores = evaluate_report(report)
        scores_md = format_scores_markdown(scores)
    except Exception:
        scores_md = "（评估模块暂时不可用）"

    # 原始JSON
    raw_json = json.dumps(report, ensure_ascii=False, indent=2)

    # 导出文件
    export_file = _make_export_file(report)

    progress = log("\n🎉 **分析全部完成！** 可在右侧查看报告、评估结果和原始JSON")
    yield report_md, progress, scores_md, raw_json, export_file


def _make_export_file(report: dict):
    """生成可下载的 JSON 文件"""
    try:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json",
            encoding="utf-8", delete=False
        )
        json.dump(report, tmp, ensure_ascii=False, indent=2)
        tmp.close()
        return tmp.name
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════
#  Gradio 界面
# ════════════════════════════════════════════════════════════════

def create_ui():
    with gr.Blocks(title="DK-SAR 化工安全分析系统") as demo:

        gr.HTML("""
        <div style="text-align:center; padding:24px 0 8px 0;">
          <h1 style="font-size:28px; margin-bottom:4px;">
            🏭 DK-SAR 化工安全分析系统
          </h1>
          <p style="color:#555; margin:0;">
            双知识增强自适应推理 &nbsp;·&nbsp; LangGraph 多智能体编排
            &nbsp;·&nbsp; 自动化 HAZOP 分析
          </p>
          <p style="color:#999; font-size:12px; margin-top:4px;">
            三智能体流水线：上下文解析 → RAG增强生成 → 物理反思校验（最多3轮修正）
          </p>
        </div>
        """)

        with gr.Row():
            # ── 左列：输入 ────────────────────────────────────
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 📝 输入化工异常场景")
                user_input = gr.Textbox(
                    label="场景描述",
                    placeholder="例如：反应釜温度从80°C升至120°C，搅拌器仍在运行...",
                    lines=5,
                )
                analyze_btn = gr.Button(
                    "🚀 开始分析", variant="primary", size="lg"
                )
                gr.Markdown("**💡 示例场景（点击加载）：**")
                for i, ex in enumerate(EXAMPLES):
                    b = gr.Button(
                        f"示例{i+1}：{ex[:35]}…",
                        size="sm", variant="secondary"
                    )
                    b.click(fn=lambda e=ex: e, outputs=user_input)

            # ── 右列：输出 ────────────────────────────────────
            with gr.Column(scale=2):
                with gr.Tabs():

                    with gr.Tab("📋 分析报告"):
                        report_output = gr.Markdown()

                    with gr.Tab("⚙️ 执行进度"):
                        progress_output = gr.Markdown(
                            value="点击「开始分析」后，这里会实时显示每个智能体的执行状态..."
                        )

                    with gr.Tab("📊 论文指标评估"):
                        scores_output = gr.Markdown(
                            value="分析完成后自动计算 PCC / CCC / RDI / LCC 四项指标..."
                        )

                    with gr.Tab("📄 原始 JSON"):
                        json_output = gr.Code(language="json")

                gr.Markdown("**📥 导出报告：**")
                export_btn = gr.File(
                    label="下载 JSON 报告",
                    interactive=False,
                )

        # 绑定事件（streaming 模式）
        analyze_btn.click(
            fn=analyze_streaming,
            inputs=[user_input],
            outputs=[report_output, progress_output, scores_output,
                     json_output, export_btn],
        )

        gr.Markdown("""
        ---
        **架构**：LangGraph 状态机 | 三节点（Context → RAG ⇄ Reflection）| 条件路由修正循环

        **知识库**：正样本库（CSB事故案例）| 负样本库（物理谬误）| IEC 61882 Schema

        **评估指标**：PCC 物理概念覆盖率 | CCC 因果链完整性 | RDI 建议详细度 | LCC 案例关联度
        """)

    return demo


# ════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DK-SAR 化工安全分析系统 v2  启动中...")
    print("="*60)

    initialize_system()

    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
