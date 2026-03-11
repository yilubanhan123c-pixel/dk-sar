"""
DK-SAR 网页界面 v4
改进：适配新版分层报告结构，增加摘要卡片，展示物理校验规则细节
"""
import os
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json, tempfile
import gradio as gr

from main import initialize_system
from agents.context_agent import ContextAgent
from agents.rag_agent import RAGAgent
from agents.reflection_agent import ReflectionAgent
from eval import evaluate_report, format_scores_markdown
from feedback import save_feedback, get_stats, format_stats_markdown
import config

EXAMPLES = [
    "反应釜温度从80°C升至120°C，搅拌器仍在运行，冷却水流量正常",
    "换热器出口温度持续升高，超过设计温度15°C，操作人员停止了搅拌",
    "储罐液位计显示液位正常，但进料流量持续大于出料流量，差值约20%",
    "管道发生少量可燃气体泄漏，现场人员认为气体会自然扩散，未采取措施",
    "反应器压力上升，泄压阀自动开启，操作人员判断系统已经安全继续操作",
]

_current_result = {"report": None, "input": "", "scores": None}


def render_report(report: dict) -> str:
    if not report:
        return "❌ 报告生成失败"
    lines = []
    node = report.get("node_info", {})

    # ── 顶部摘要卡片 ──────────────────────────────────────────
    summary = report.get("summary", {})
    if summary:
        lines.append("## 📌 分析摘要")
        lines.append(f"> **核心偏差**：{summary.get('core_deviation', 'N/A')}")
        lines.append(f"> \n> **首要怀疑**：{summary.get('top_suspect', 'N/A')}")
        lines.append(f"> \n> **最高风险**：{summary.get('highest_risk', 'N/A')}")
        lines.append(f"> \n> 🚨 **首要动作**：{summary.get('immediate_action', 'N/A')}")
        lines.append("")

    # ── 基本信息 ───────────────────────────────────────────────
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

        # ── 分层原因 ────────────────────────────────────────────
        causes = deviation.get("causes", {})

        # 兼容旧格式（list）和新格式（dict分层）
        if isinstance(causes, list):
            lines.append("\n### 🔍 可能原因")
            for i, c in enumerate(causes, 1):
                lines.append(f"{i}. **[{c.get('type','')}]** {c.get('description','')}")
        else:
            lines.append("\n### 🔍 原因分析")

            primary = causes.get("primary", [])
            if primary:
                lines.append("\n**🔴 首要怀疑**（最可能，优先排查）")
                for c in primary:
                    lines.append(f"- **[{c.get('type','')}]** {c.get('description','')}")

            secondary = causes.get("secondary", [])
            if secondary:
                lines.append("\n**🟡 次要怀疑**（可能性较低，次优先）")
                for c in secondary:
                    lines.append(f"- **[{c.get('type','')}]** {c.get('description','')}")

            pending = causes.get("pending", [])
            if pending:
                lines.append("\n**🔵 待验证**（需现场核实）")
                for c in pending:
                    lines.append(f"- {c.get('description','')}")

        # ── 后果链 ─────────────────────────────────────────────
        lines.append("\n### ⚠️ 后果链")
        consequences = deviation.get("consequences", [])
        for c in consequences:
            if isinstance(c, dict):
                stage = c.get("stage", "")
                desc = c.get("description", "")
                icon = {"直接后果": "①", "升级后果": "→", "极端后果": "⚠️"}.get(stage, "→")
                lines.append(f"{icon} **{stage}**：{desc}")
            else:
                lines.append(f"→ {c}")

        # ── 保护措施 ────────────────────────────────────────────
        lines.append("\n### 🛡️ 现有保护措施")
        for sg in deviation.get("safeguards", []):
            eff = sg.get('effectiveness', '')
            icon = "✅" if eff == "有效" else ("⚠️" if eff == "部分有效" else "❌")
            reason = sg.get('effectiveness_reason', '')
            reason_text = f"（{reason}）" if reason else f"（{eff}）"
            lines.append(f"- {icon} **{sg.get('measure','')}** {reason_text}")

        # ── 建议措施 ────────────────────────────────────────────
        lines.append("\n### 💡 建议措施")
        for rec in deviation.get("recommendations", []):
            p = rec.get('priority', '')
            icon = "🔴" if p == "高" else ("🟡" if p == "中" else "🟢")
            lines.append(f"- {icon} **[{p}]** {rec.get('action','')}")

    # ── 元数据 ─────────────────────────────────────────────────
    lines.append("\n---")
    meta = report.get("analysis_metadata", {})
    lines.append("\n### 📊 分析元数据")
    ref = meta.get("referenced_cases", [])
    if ref:
        lines.append(f"- **参考案例**: {', '.join(ref)}")
    lines.append(f"- **物理反思轮次**: {meta.get('reflection_rounds', 0)} 轮")

    confidence = meta.get("confidence_level", "N/A")
    confidence_reason = meta.get("confidence_reason", "")
    conf_text = f"{confidence}（{confidence_reason}）" if confidence_reason else confidence
    lines.append(f"- **分析置信度**: {conf_text}")

    issues = meta.get("physical_issues_found", [])
    if issues:
        lines.append(f"- **物理校验**: 已修正 {len(issues)} 个问题")
        for iss in issues[:3]:
            lines.append(f"  - {iss[:60]}...")
    else:
        lines.append("- **物理校验**: ✅ 通过（质量守恒 / 能量守恒 / 时序因果 均无异常）")

    return "\n".join(lines)


def analyze_streaming(user_input: str):
    global _current_result
    if not user_input or not user_input.strip():
        yield "⚠️ 请输入场景描述", "", "", "", None
        return

    _current_result = {"report": None, "input": user_input, "scores": None}
    progress_log = []
    report = None
    context = None

    def log(msg):
        progress_log.append(msg)
        return "\n".join(progress_log)

    # 节点1
    progress = log("⏳ **[节点1/3]** 上下文智能体 — 解析场景中...")
    yield "分析中，请稍候...", progress, "", "", None
    try:
        ctx_result = ContextAgent().run(user_input.strip())
        context = ctx_result["context"]
        progress = log(
            f"✅ **节点1 完成** — {context['equipment']} / "
            f"{context['parameter']} / {context['deviation_direction']}"
        )
    except Exception as e:
        yield "❌ 节点1失败", log(f"❌ {e}"), "", "", None
        return
    yield "分析中，请稍候...", progress, "", "", None

    # 节点2+3循环
    correction_guidance = ""
    round_num = 0
    while round_num < config.MAX_REFLECTION_ROUNDS:
        progress = log(f"\n⏳ **[节点2/3]** RAG分析智能体 — 第{round_num+1}轮生成...")
        yield "分析中，请稍候...", progress, "", "", None
        try:
            report = RAGAgent().run(
                context=context,
                correction_guidance=correction_guidance,
                reflection_rounds=round_num
            )
            ref = report.get("analysis_metadata", {}).get("referenced_cases", [])
            progress = log(f"✅ **节点2 完成** — 参考案例: {ref}")
        except Exception as e:
            yield "❌ 节点2失败", log(f"❌ {e}"), "", "", None
            return
        yield "分析中，请稍候...", progress, "", "", None

        progress = log(f"\n⏳ **[节点3/3]** 物理反思智能体 — 第{round_num+1}轮校验...")
        progress += "\n正在检查：质量守恒 / 能量守恒 / 动量守恒 / 时序因果 / 数值合理性..."
        yield "分析中，请稍候...", progress, "", "", None
        try:
            ref_result = ReflectionAgent().run(report)
            if ref_result["passed"]:
                progress = log(
                    "✅ **节点3 通过** — 五项物理规则均无异常\n"
                    "  ✓ 质量守恒  ✓ 能量守恒  ✓ 动量守恒  ✓ 时序因果  ✓ 数值合理性"
                )
                yield "分析中，请稍候...", progress, "", "", None
                break
            else:
                n = len(ref_result["issues"])
                issue_lines = "\n".join(
                    f"  - [{i.get('issue_type','')}] {i['description'][:55]}..."
                    for i in ref_result["issues"][:3]
                )
                progress = log(
                    f"⚠️ **节点3 发现{n}个物理问题** — 返回节点2修正\n{issue_lines}"
                )
                correction_guidance = ref_result["correction_guidance"]
                round_num += 1
                if round_num >= config.MAX_REFLECTION_ROUNDS:
                    progress = log(f"⚠️ 已达最大轮次{config.MAX_REFLECTION_ROUNDS}，使用当前版本")
                yield "分析中，请稍候...", progress, "", "", None
        except Exception as e:
            progress = log(f"❌ 节点3失败: {e}")
            break

    if not report:
        yield "❌ 分析失败", progress, "", "", None
        return

    if "analysis_metadata" not in report:
        report["analysis_metadata"] = {}
    report["analysis_metadata"]["reflection_rounds"] = round_num + 1

    report_md = render_report(report)
    try:
        scores = evaluate_report(report)
        scores_md = format_scores_markdown(scores)
    except Exception:
        scores, scores_md = None, "（评估暂不可用）"

    _current_result = {"report": report, "input": user_input, "scores": scores}
    raw_json = json.dumps(report, ensure_ascii=False, indent=2)
    export_file = _make_export_file(report)
    progress = log("\n🎉 **分析完成！** 请在右侧查看报告，并为本次分析评分 👇")
    yield report_md, progress, scores_md, raw_json, export_file


def submit_feedback(rating: int, comment: str):
    global _current_result
    if not _current_result.get("report"):
        return "⚠️ 请先完成一次分析，再提交评分"
    try:
        save_feedback(
            user_input=_current_result["input"],
            report=_current_result["report"],
            rating=rating,
            comment=comment,
            scores=_current_result.get("scores"),
        )
        return f"✅ 感谢反馈！已记录 {'⭐' * rating} 评分"
    except Exception as e:
        return f"❌ 记录失败: {e}"


def refresh_stats():
    return format_stats_markdown(get_stats())


def _make_export_file(report):
    try:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8", delete=False
        )
        json.dump(report, tmp, ensure_ascii=False, indent=2)
        tmp.close()
        return tmp.name
    except Exception:
        return None


def create_ui():
    with gr.Blocks(title="DK-SAR 化工安全分析系统") as demo:

        gr.HTML("""
        <div style="text-align:center;padding:24px 0 8px 0;">
          <h1 style="font-size:26px;margin-bottom:4px;">🏭 DK-SAR 化工安全分析系统</h1>
          <p style="color:#555;margin:0;">
            双知识增强自适应推理 · LangGraph 多智能体编排 · 自动化 HAZOP 分析
          </p>
          <p style="color:#999;font-size:12px;margin-top:4px;">
            三节点：上下文解析 → RAG增强生成 → 物理反思校验（最多3轮修正）
          </p>
        </div>""")

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### 📝 输入化工异常场景")
                user_input = gr.Textbox(
                    label="场景描述",
                    placeholder="例如：反应釜温度从80°C升至120°C...",
                    lines=5
                )
                analyze_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")
                gr.Markdown("**💡 示例场景：**")
                for i, ex in enumerate(EXAMPLES):
                    b = gr.Button(f"示例{i+1}：{ex[:32]}…", size="sm", variant="secondary")
                    b.click(fn=lambda e=ex: e, outputs=user_input)

                gr.Markdown("---\n### ⭐ 为本次分析评分")
                rating_slider = gr.Slider(minimum=1, maximum=5, step=1, value=4,
                                          label="评分（1-5星）")
                comment_box = gr.Textbox(
                    label="反馈意见（可选）",
                    placeholder="分析结果是否准确？建议如何改进？",
                    lines=2
                )
                feedback_btn = gr.Button("提交评分", variant="secondary")
                feedback_result = gr.Markdown()

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📋 分析报告"):
                        report_output = gr.Markdown()
                    with gr.Tab("⚙️ 执行进度"):
                        progress_output = gr.Markdown(
                            value="点击「开始分析」后，这里实时显示每个智能体的执行状态..."
                        )
                    with gr.Tab("📊 论文指标评估"):
                        scores_output = gr.Markdown(
                            value="分析完成后自动计算 PCC / CCC / RDI / LCC 四项指标..."
                        )
                    with gr.Tab("📈 使用数据统计"):
                        stats_output = gr.Markdown(value=format_stats_markdown(get_stats()))
                        refresh_btn = gr.Button("🔄 刷新统计", size="sm")
                    with gr.Tab("📄 原始 JSON"):
                        json_output = gr.Code(language="json")

                gr.Markdown("**📥 导出报告：**")
                export_btn = gr.File(label="下载 JSON 报告", interactive=False)

        analyze_btn.click(
            fn=analyze_streaming, inputs=[user_input],
            outputs=[report_output, progress_output, scores_output, json_output, export_btn]
        )
        feedback_btn.click(
            fn=submit_feedback, inputs=[rating_slider, comment_box],
            outputs=[feedback_result]
        )
        refresh_btn.click(fn=refresh_stats, outputs=[stats_output])

        gr.Markdown("""---
**架构**：LangGraph 状态机 | 三节点条件路由 | 最多3轮物理反思修正循环

**知识库**：正样本库（36个CSB事故案例）| 负样本库（55条物理谬误）| IEC 61882 Schema

**校验规则**：质量守恒 · 能量守恒 · 动量守恒 · 时序因果 · 数值合理性""")

    return demo


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DK-SAR 化工安全分析系统 v4  启动中...")
    print("="*60)
    initialize_system()
    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1", server_port=7860,
        share=False, inbrowser=True, theme=gr.themes.Soft()
    )
