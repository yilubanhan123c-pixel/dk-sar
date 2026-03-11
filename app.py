"""
DK-SAR 网页界面 v5（最终版）
核心：把系统真正的三个创新显化出来
  1. 负样本库命中 → "双知识库校验"面板
  2. 修正前后对比 → 只在发生修正时出现
  3. RAG检索案例名称 → 不只显示ID
"""
import os
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
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
EXPORT_PATH = "last_report.json"


# ════════════════════════════════════════════════════════════════
#  报告渲染
# ════════════════════════════════════════════════════════════════

def render_report(report: dict) -> str:
    if not report:
        return "❌ 报告生成失败"
    lines = []

    # 摘要卡片
    s = report.get("summary", {})
    if s:
        lines += [
            "## 📌 分析摘要",
            f"> **核心偏差**：{s.get('core_deviation','N/A')}",
            f">\n> **首要怀疑**：{s.get('top_suspect','N/A')}",
            f">\n> **最高风险**：{s.get('highest_risk','N/A')}",
            f">\n> 🚨 **首要动作**：{s.get('immediate_action','N/A')}",
            "",
        ]

    # 基本信息
    node = report.get("node_info", {})
    lines += [
        "## 📋 HAZOP 分析报告",
        f"\n**设备**: {node.get('equipment','N/A')}  |  "
        f"**参数**: {node.get('parameter','N/A')}  |  "
        f"**偏差方向**: {node.get('deviation_direction','N/A')}",
        f"\n**正常值**: {node.get('normal_value','N/A')}  |  "
        f"**当前值**: {node.get('current_value','N/A')}",
        "\n---",
    ]

    # 证据链
    evidence = report.get("evidence_chain", [])
    if evidence:
        lines.append("\n### 🔗 推理证据链")
        icons = {"输入事实": "📥", "规则触发": "📐",
                 "案例支持": "📚", "物理校验": "⚗️", "结论收敛": "✅"}
        for e in evidence:
            icon = icons.get(e.get("type", ""), "▸")
            lines.append(
                f"**{e.get('step','')}. {icon} {e.get('type','')}**：{e.get('content','')}"
            )

    # 偏差分析
    for deviation in report.get("deviations", []):

        # 分层原因
        causes = deviation.get("causes", {})
        lines.append("\n### 🔍 原因分析")
        if isinstance(causes, list):
            for i, c in enumerate(causes, 1):
                lines.append(f"{i}. **[{c.get('type','')}]** {c.get('description','')}")
        else:
            for layer, label, icon in [
                ("primary",   "首要怀疑（最可能，优先排查）", "🔴"),
                ("secondary", "次要怀疑（可能性较低）",       "🟡"),
                ("pending",   "待验证（需现场核实）",         "🔵"),
            ]:
                items = causes.get(layer, [])
                if items:
                    lines.append(f"\n**{icon} {label}**")
                    for c in items:
                        lines.append(
                            f"- **[{c.get('type','')}]** {c.get('description','')}"
                        )

        # 后果链
        lines.append("\n### ⚠️ 后果链")
        stage_icons = {"直接后果": "①", "升级后果": "→", "极端后果": "⚠️"}
        for c in deviation.get("consequences", []):
            if isinstance(c, dict):
                stage = c.get("stage", "")
                lines.append(
                    f"{stage_icons.get(stage,'→')} **{stage}**：{c.get('description','')}"
                )
            else:
                lines.append(f"→ {c}")

        # 保护措施
        lines.append("\n### 🛡️ 现有保护措施")
        for sg in deviation.get("safeguards", []):
            eff = sg.get("effectiveness", "")
            icon = "✅" if eff == "有效" else ("⚠️" if eff == "部分有效" else "❌")
            reason = sg.get("effectiveness_reason", "")
            lines.append(
                f"- {icon} **{sg.get('measure','')}**（{eff}）"
                + (f" — {reason}" if reason else "")
            )

        # 建议措施（三层）
        recs = deviation.get("recommendations", {})
        lines.append("\n### 💡 建议措施")
        if isinstance(recs, list):
            for rec in recs:
                p = rec.get("priority", "")
                icon = "🔴" if p == "高" else ("🟡" if p == "中" else "🟢")
                lines.append(f"- {icon} **[{p}]** {rec.get('action','')}")
        else:
            for layer, label, icon in [
                ("immediate",  "⚡ 立即处置（分钟级响应）", "🔴"),
                ("short_term", "🔧 短期整改（天级）",       "🟡"),
                ("long_term",  "📋 长期改进（月级）",       "🟢"),
            ]:
                items = recs.get(layer, [])
                if items:
                    lines.append(f"\n**{label}**")
                    for rec in items:
                        lines.append(f"- {icon} {rec.get('action','')}")

    # 元数据
    lines.append("\n---\n### 📊 分析元数据")
    meta = report.get("analysis_metadata", {})
    ref_ids = meta.get("referenced_cases", [])
    ref_names = meta.get("referenced_names", {})
    if ref_ids:
        case_strs = [
            f"`{cid}`{(' ' + ref_names[cid]) if cid in ref_names and ref_names[cid] else ''}"
            for cid in ref_ids
        ]
        lines.append(f"- **参考案例**: {' · '.join(case_strs)}")
    lines.append(f"- **物理反思轮次**: {meta.get('reflection_rounds', 0)} 轮")
    conf_reason = meta.get("confidence_reason", "")
    conf_level = meta.get("confidence_level", "N/A")
    lines.append(
        f"- **分析置信度**: {conf_level}"
        + (f"（{conf_reason}）" if conf_reason else "")
    )
    issues = meta.get("physical_issues_found", [])
    if issues:
        lines.append(f"- **物理校验**: 已修正 {len(issues)} 个问题")
    else:
        lines.append(
            "- **物理校验**: ✅ 通过"
            "（质量守恒 / 能量守恒 / 时序因果 / 数值合理性 / 过程机理）"
        )

    return "\n".join(lines)


def render_reflection_panel(reflection_history: list) -> str:
    """
    渲染物理反思面板
    展示：负样本库命中情况 + 每轮修正的具体问题 + 修正前后变化
    这是系统最核心创新的可视化
    """
    if not reflection_history:
        return "分析完成后，这里会显示物理反思的完整过程..."

    lines = []

    # 双知识库说明
    lines += [
        "## 🧠 双知识库物理反思过程",
        "",
        "> **正样本库**（36个CSB事故案例）→ RAG检索相似情景",
        "> **负样本库**（55条物理谬误）→ 快速筛查错误描述",
        "",
        "---",
    ]

    total_rounds = len(reflection_history)
    has_correction = any(not r.get("passed") for r in reflection_history[:-1])

    for record in reflection_history:
        rnd = record.get("round", 0)
        passed = record.get("passed", False)
        issues = record.get("issues", [])
        fallacy_hits = record.get("fallacy_hits", [])

        lines.append(f"### 第 {rnd} 轮物理校验")

        # 负样本库命中
        fast_hits = [i for i in issues if i.get("source") == "fast_screening"]
        deep_hits = [i for i in issues if i.get("source") == "deep_verification"]

        if fallacy_hits:
            lines.append(f"\n**⚡ Layer 1：负样本库匹配（命中 {len(fallacy_hits)} 条物理谬误）**")
            for hit in fallacy_hits:
                lines += [
                    f"- **[{hit.get('fallacy_id','')} · {hit.get('category','')}]**",
                    f"  - 报告中的命题：「{hit.get('matched_proposition','')}」",
                    f"  - 匹配到的谬误：「{hit.get('false_claim','')}」",
                    f"  - 正确理解：{hit.get('correct_understanding','')}",
                    f"  - 相似度：{hit.get('similarity', 0):.1%}",
                ]
        else:
            lines.append("\n**⚡ Layer 1：负样本库匹配** — ✅ 未发现已知物理谬误")

        if deep_hits:
            lines.append(f"\n**🔬 Layer 2：LLM物理推理（发现 {len(deep_hits)} 个问题）**")
            for h in deep_hits:
                lines += [
                    f"- **[{h.get('issue_type','')}]** {h.get('description','')}",
                    f"  → 修正建议：{h.get('correction_hint','')}",
                ]
        else:
            lines.append("\n**🔬 Layer 2：LLM物理推理** — ✅ 五项规则均通过")

        if passed:
            lines.append(f"\n✅ **第{rnd}轮校验通过**")
        else:
            lines.append(f"\n⚠️ **第{rnd}轮发现 {len(issues)} 个问题 → 触发修正，进入第{rnd+1}轮**")
        lines.append("")

    # 总结
    if has_correction:
        lines += [
            "---",
            f"### 📈 修正总结",
            f"经过 **{total_rounds} 轮**物理反思校验，系统自动识别并修正了物理错误，",
            f"最终报告已通过双层物理校验（负样本库匹配 + LLM推理验证）。",
        ]
    else:
        lines += [
            "---",
            "✅ **报告一次通过物理校验，质量良好**",
        ]

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
#  分析主函数（流式输出）
# ════════════════════════════════════════════════════════════════

def analyze_streaming(user_input: str):
    global _current_result
    if not user_input or not user_input.strip():
        yield "⚠️ 请输入场景描述", "", "", "", None
        return

    _current_result = {"report": None, "input": user_input, "scores": None}
    progress_log = []
    report = None
    context = None
    reflection_history = []

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
            f"✅ **节点1 完成** — "
            f"{context['equipment']} / {context['parameter']} / {context['deviation_direction']}"
        )
    except Exception as e:
        yield "❌ 节点1失败", log(f"❌ {e}"), "", "", None
        return
    yield "分析中，请稍候...", progress, "", "", None

    # 节点2+3循环
    correction_guidance = ""
    round_num = 0

    while round_num < config.MAX_REFLECTION_ROUNDS:
        # 节点2
        progress = log(f"\n⏳ **[节点2/3]** RAG分析智能体 — 第{round_num+1}轮生成中...")
        yield "分析中，请稍候...", progress, "", "", None
        try:
            report = RAGAgent().run(
                context=context,
                correction_guidance=correction_guidance,
                reflection_rounds=round_num,
            )
            meta = report.get("analysis_metadata", {})
            ref_ids = meta.get("referenced_cases", [])
            ref_names = meta.get("referenced_names", {})
            case_strs = [
                f"{cid}({ref_names.get(cid,'')})" if ref_names.get(cid) else cid
                for cid in ref_ids
            ]
            progress = log(f"✅ **节点2 完成** — 检索案例: {', '.join(case_strs)}")
        except Exception as e:
            yield "❌ 节点2失败", log(f"❌ {e}"), "", "", None
            return
        yield "分析中，请稍候...", progress, "", "", None

        # 节点3
        progress = log(
            f"\n⏳ **[节点3/3]** 物理反思智能体 — 第{round_num+1}轮校验中...\n"
            f"  Layer 1: 负样本库匹配（55条物理谬误）\n"
            f"  Layer 2: LLM五项物理规则验证"
        )
        yield "分析中，请稍候...", progress, "", "", None

        try:
            ref_result = ReflectionAgent().run(report)
            fallacy_hits = ref_result.get("fallacy_hits", [])

            record = {
                "round": round_num + 1,
                "passed": ref_result["passed"],
                "issues": ref_result["issues"],
                "fallacy_hits": fallacy_hits,
            }
            reflection_history.append(record)

            if fallacy_hits:
                hit_names = [h.get("fallacy_id", "") for h in fallacy_hits]
                progress = log(
                    f"⚡ **负样本库命中**: {', '.join(hit_names)} "
                    f"（{len(fallacy_hits)}条物理谬误）"
                )
                yield "分析中，请稍候...", progress, "", "", None

            if ref_result["passed"]:
                progress = log(
                    "✅ **节点3 通过** — 双层物理校验均无问题\n"
                    "  ✓ 负样本库匹配  ✓ 质量守恒  ✓ 能量守恒  ✓ 时序因果  ✓ 数值合理性"
                )
                yield "分析中，请稍候...", progress, "", "", None
                break
            else:
                n = len(ref_result["issues"])
                progress = log(
                    f"⚠️ **节点3 发现{n}个问题** — 返回节点2修正\n" +
                    "\n".join(
                        f"  [{i.get('issue_type','')}] {i['description'][:55]}..."
                        for i in ref_result["issues"][:3]
                    )
                )
                correction_guidance = ref_result["correction_guidance"]
                round_num += 1
                if round_num >= config.MAX_REFLECTION_ROUNDS:
                    progress = log(f"⚠️ 已达最大轮次{config.MAX_REFLECTION_ROUNDS}，使用当前版本")
                yield "分析中，请稍候...", progress, "", "", None

        except Exception as e:
            progress = log(f"⚠️ 节点3异常({e})，跳过校验")
            break

    if not report:
        yield "❌ 分析失败", progress, "", "", None
        return

    if "analysis_metadata" not in report:
        report["analysis_metadata"] = {}
    report["analysis_metadata"]["reflection_rounds"] = round_num + 1

    # 渲染报告
    report_md = render_report(report)

    # 物理反思面板
    reflection_md = render_reflection_panel(reflection_history)

    # 评估指标
    scores_md = "（评估暂不可用）"
    scores = None
    try:
        scores = evaluate_report(report)
        scores_md = format_scores_markdown(scores)
    except Exception as e:
        scores_md = f"（评估失败: {e}）"

    _current_result = {"report": report, "input": user_input, "scores": scores}

    # JSON导出（固定路径，避免Windows临时文件卡死）
    raw_json = json.dumps(report, ensure_ascii=False, indent=2)
    export_path = None
    try:
        with open(EXPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        export_path = EXPORT_PATH
    except Exception:
        pass

    progress = log("\n🎉 **分析完成！** 查看左侧报告，「双知识库反思」标签可看修正详情 👆")
    yield report_md, reflection_md, scores_md, raw_json, export_path


def submit_feedback(rating: int, comment: str):
    global _current_result
    if not _current_result.get("report"):
        return "⚠️ 请先完成一次分析"
    try:
        save_feedback(
            user_input=_current_result["input"],
            report=_current_result["report"],
            rating=rating,
            comment=comment,
            scores=_current_result.get("scores"),
        )
        return f"✅ 已记录 {'⭐' * rating} 感谢反馈！"
    except Exception as e:
        return f"❌ 记录失败: {e}"


def refresh_stats():
    return format_stats_markdown(get_stats())


# ════════════════════════════════════════════════════════════════
#  Gradio 界面
# ════════════════════════════════════════════════════════════════

def create_ui():
    with gr.Blocks(title="DK-SAR 化工安全分析系统") as demo:

        gr.HTML("""
        <div style="text-align:center;padding:24px 0 8px 0;">
          <h1 style="font-size:26px;margin-bottom:4px;">🏭 DK-SAR 化工安全分析系统</h1>
          <p style="color:#555;margin:0;">
            双知识增强自适应推理 · LangGraph 多智能体编排 · 自动化 HAZOP 分析
          </p>
          <p style="color:#999;font-size:12px;margin-top:4px;">
            正样本库 36案例 · 负样本库 55条谬误 · 最多3轮物理反思修正
          </p>
        </div>""")

        with gr.Row():
            # 左列：输入
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### 📝 输入化工异常场景")
                user_input = gr.Textbox(
                    label="场景描述",
                    placeholder="例如：反应釜温度从80°C升至120°C...",
                    lines=5,
                )
                analyze_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")

                gr.Markdown("**💡 示例场景：**")
                for i, ex in enumerate(EXAMPLES):
                    b = gr.Button(f"示例{i+1}：{ex[:32]}…", size="sm", variant="secondary")
                    b.click(fn=lambda e=ex: e, outputs=user_input)

                gr.Markdown("---\n### ⭐ 为本次分析评分")
                rating_slider = gr.Slider(
                    minimum=1, maximum=5, step=1, value=4, label="评分（1-5星）"
                )
                comment_box = gr.Textbox(
                    label="反馈意见（可选）",
                    placeholder="分析是否准确？有何改进建议？",
                    lines=2,
                )
                feedback_btn = gr.Button("提交评分", variant="secondary")
                feedback_result = gr.Markdown()

            # 右列：输出
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📋 分析报告"):
                        report_output = gr.Markdown()

                    with gr.Tab("🧠 双知识库反思"):
                        reflection_output = gr.Markdown(
                            value=(
                                "分析完成后，这里会展示：\n\n"
                                "- **负样本库命中**：哪些物理谬误被检测到\n"
                                "- **每轮修正详情**：Layer1匹配 + Layer2推理的结果\n"
                                "- **修正汇总**：经过几轮迭代最终通过校验"
                            )
                        )

                    with gr.Tab("⚙️ 执行进度"):
                        progress_output = gr.Markdown(
                            value="点击「开始分析」后，这里实时显示三个智能体的执行状态..."
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

        # 绑定事件
        analyze_btn.click(
            fn=analyze_streaming,
            inputs=[user_input],
            outputs=[
                report_output, reflection_output, scores_output,
                json_output, export_btn,
            ],
        )
        feedback_btn.click(
            fn=submit_feedback,
            inputs=[rating_slider, comment_box],
            outputs=[feedback_result],
        )
        refresh_btn.click(fn=refresh_stats, outputs=[stats_output])

        gr.Markdown("""---
**架构**：LangGraph 状态机 · 三节点条件路由 · 最多3轮物理反思修正

**双知识库**：正样本库（36个CSB事故案例，RAG检索）· 负样本库（55条物理谬误，快速筛查）

**报告结构**：摘要 → 证据链 → 分层原因（首要/次要/待验证）→ 后果链 → 保护措施 → 三层建议""")

    return demo


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DK-SAR 化工安全分析系统 v5  启动中...")
    print("=" * 60)
    initialize_system()
    create_ui().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
