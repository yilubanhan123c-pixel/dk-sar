import base64
import json
import mimetypes
import os
from pathlib import Path

import gradio as gr

import config
from agents.context_agent import ContextAgent
from agents.rag_agent import RAGAgent
from agents.reflection_agent import ReflectionAgent
from eval import evaluate_report, format_scores_markdown
from feedback import format_stats_markdown, get_stats, save_feedback
from main import initialize_system

os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

EXAMPLES = [
    ["反应釜温度从80°C升至120°C，搅拌器仍在运行，冷却水流量正常。"],
    ["换热器出口温度持续升高，超过设计温度15°C，操作人员停止了搅拌。"],
    ["储罐液位计显示正常，但进料流量持续大于出料流量，差值约20%。"],
    ["管道发生少量可燃气体泄漏，现场人员认为气体会自然扩散，未采取措施。"],
    ["反应器压力上升，泄压阀自动开启，操作人员判断系统已经安全继续操作。"],
]

TEXT_TAB_EXAMPLES = [
    "反应釜热失控",
    "液氨中毒窒息",
    "储罐泄漏扩散",
    "换热器结垢堵塞",
    "安全阀超压排放",
]

IMAGE_SYSTEM_PROMPT = """你是一名资深化工安全工程师和 HAZOP 分析专家。
请基于用户上传的现场照片，输出结构化中文分析，要求专业但易懂，适合一线巡检人员阅读。

请按以下结构输出：
1. 风险等级：仅可使用 高风险 / 中风险 / 低风险
2. 风险识别：逐条列出照片中观察到的异常、隐患或不安全行为
3. 判断依据：说明你为什么这样判断，可引用设备状态、环境特征、PPE、泄漏迹象、腐蚀、标识缺失等
4. 建议措施：给出现场可执行的处置建议，优先给出立即措施
5. HAZOP 关联：如果能关联到 HAZOP 偏差，请指出可能的参数与偏差方向，例如 温度偏高、压力偏高、流量偏低、液位偏高、泄漏、成分异常

如果照片信息不足，请明确说明“不足以确认”的部分，不要编造细节。"""

_current_result = {
    "report": None,
    "input": "",
    "scores": None,
    "reflection_history": [],
    "image_seed": "",
}
EXPORT_PATH = "last_report.json"

THEME = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="emerald",
    neutral_hue="slate",
)


def render_report(report: dict) -> str:
    if not report:
        return "请先在左侧输入场景并开始分析。"

    lines = []
    summary = report.get("summary", {})
    node = report.get("node_info", {})

    if summary:
        lines.extend(
            [
                "## 分析摘要",
                f"- 核心偏差：{summary.get('core_deviation', 'N/A')}",
                f"- 首要怀疑：{summary.get('top_suspect', 'N/A')}",
                f"- 最高风险：{summary.get('highest_risk', 'N/A')}",
                f"- 首要动作：{summary.get('immediate_action', 'N/A')}",
                "",
            ]
        )

    lines.extend(
        [
            "## HAZOP 报告",
            f"- 设备：{node.get('equipment', 'N/A')}",
            f"- 参数：{node.get('parameter', 'N/A')}",
            f"- 偏差方向：{node.get('deviation_direction', 'N/A')}",
            f"- 正常值：{node.get('normal_value', 'N/A')}",
            f"- 当前值：{node.get('current_value', 'N/A')}",
            "",
        ]
    )

    for index, deviation in enumerate(report.get("deviations", []), start=1):
        lines.append(f"### 偏差条目 {index}")

        causes = deviation.get("causes", {})
        if isinstance(causes, dict):
            for key, title in [
                ("primary", "首要怀疑"),
                ("secondary", "次要怀疑"),
                ("pending", "待验证项"),
            ]:
                items = causes.get(key, [])
                if items:
                    lines.append(f"**{title}**")
                    for item in items:
                        lines.append(
                            f"- [{item.get('type', '未分类')}] {item.get('description', '')}"
                        )

        consequences = deviation.get("consequences", [])
        if consequences:
            lines.append("**后果链**")
            for item in consequences:
                if isinstance(item, dict):
                    lines.append(
                        f"- {item.get('stage', '后果')}：{item.get('description', '')}"
                    )
                else:
                    lines.append(f"- {item}")

        safeguards = deviation.get("safeguards", [])
        if safeguards:
            lines.append("**现有保护措施**")
            for safeguard in safeguards:
                lines.append(
                    f"- {safeguard.get('measure', '')}："
                    f"{safeguard.get('effectiveness', '未说明')}，"
                    f"{safeguard.get('effectiveness_reason', '暂无说明')}"
                )

        recommendations = deviation.get("recommendations", {})
        if isinstance(recommendations, dict):
            lines.append("**建议措施**")
            for key, title in [
                ("immediate", "立即处置"),
                ("short_term", "短期整改"),
                ("long_term", "长期改进"),
            ]:
                items = recommendations.get(key, [])
                if items:
                    lines.append(f"- {title}")
                    for item in items:
                        lines.append(f"  - {item.get('action', '')}")
        lines.append("")

    return "\n".join(lines).strip()


def render_sources_panel(report: dict) -> str:
    if not report:
        return "分析完成后，这里会展示参考案例、证据链和置信信息。"

    meta = report.get("analysis_metadata", {})
    evidence_chain = report.get("evidence_chain", [])
    ref_ids = meta.get("referenced_cases", [])
    ref_names = meta.get("referenced_names", {})
    issues = meta.get("physical_issues_found", [])

    lines = ["## 知识溯源"]
    if ref_ids:
        lines.append("### 参考案例")
        for case_id in ref_ids:
            case_name = ref_names.get(case_id, "")
            label = f"{case_id} - {case_name}" if case_name else case_id
            lines.append(f"- {label}")

    if evidence_chain:
        lines.append("")
        lines.append("### 证据链")
        for item in evidence_chain:
            lines.append(
                f"- Step {item.get('step', '?')} | {item.get('type', '未分类')}："
                f"{item.get('content', '')}"
            )

    lines.extend(
        [
            "",
            "### 置信与校验",
            f"- 置信水平：{meta.get('confidence_level', 'N/A')}",
            f"- 置信依据：{meta.get('confidence_reason', 'N/A')}",
            f"- 反思轮次：{meta.get('reflection_rounds', 0)}",
            f"- 物理问题数：{len(issues)}",
        ]
    )
    return "\n".join(lines).strip()


def render_reflection_panel(reflection_history: list) -> str:
    if not reflection_history:
        return "分析完成后，这里会展示反思日志和物理校验过程。"

    lines = ["## 反思日志"]
    for record in reflection_history:
        lines.append(
            f"### 第 {record.get('round', '?')} 轮 - "
            f"{'通过' if record.get('passed') else '发现问题'}"
        )

        fallacy_hits = record.get("fallacy_hits", [])
        if fallacy_hits:
            lines.append("**负样本库命中**")
            for hit in fallacy_hits:
                lines.append(
                    f"- {hit.get('fallacy_id', '')}：{hit.get('matched_proposition', '')}"
                )

        issues = record.get("issues", [])
        if issues:
            lines.append("**问题清单**")
            for issue in issues:
                lines.append(
                    f"- [{issue.get('issue_type', '未分类')}] {issue.get('description', '')}"
                )
        else:
            lines.append("- 本轮未发现新的物理问题。")
        lines.append("")

    return "\n".join(lines).strip()


def image_to_data_url(image_path: str) -> str:
    file_path = Path(image_path)
    mime_type = mimetypes.guess_type(file_path.name)[0] or "image/png"
    encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def extract_dashscope_text(response) -> str:
    if hasattr(response, "output"):
        output = response.output
    elif isinstance(response, dict):
        output = response.get("output", {})
    else:
        output = {}

    choices = getattr(output, "choices", None)
    if choices is None and isinstance(output, dict):
        choices = output.get("choices", [])
    if not choices:
        return "模型未返回可解析内容。"

    message = choices[0].get("message") if isinstance(choices[0], dict) else getattr(choices[0], "message", None)
    if not message:
        return "模型未返回可解析内容。"

    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n".join(parts)
    return str(content) if content else "模型未返回可解析内容。"


def compose_hazop_seed(image_result: str, additional_context: str) -> str:
    context_line = additional_context.strip() if additional_context else "未提供额外上下文。"
    return (
        "请基于以下现场巡检信息执行 HAZOP 文本分析：\n"
        f"补充说明：{context_line}\n"
        "现场拍照识别结果如下：\n"
        f"{image_result}"
    )


def analyze_image_risk(image_path: str, additional_context: str):
    if not image_path:
        message = "请先上传现场照片。"
        return message, "", message

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        message = "未检测到 `DASHSCOPE_API_KEY`，请先在环境变量或 `.env` 中配置后再进行图片分析。"
        return message, "", message

    try:
        from dashscope import MultiModalConversation
    except ImportError:
        message = "未安装 `dashscope`，请先执行 `pip install -r requirements.txt`。"
        return message, "", message

    user_text = "请分析这张化工现场照片中的安全风险。"
    if additional_context and additional_context.strip():
        user_text += f"\n补充说明：{additional_context.strip()}"

    try:
        data_url = image_to_data_url(image_path)
        response = MultiModalConversation.call(
            model="qwen-vl-max",
            api_key=api_key,
            messages=[
                {"role": "system", "content": [{"text": IMAGE_SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"image": data_url},
                        {"text": user_text},
                    ],
                },
            ],
        )
        result_text = extract_dashscope_text(response)
        hazop_seed = compose_hazop_seed(result_text, additional_context)
        _current_result["image_seed"] = hazop_seed
        status = "图片分析完成，可以将识别结果一键转入文本 HAZOP 分析。"
        return result_text, hazop_seed, status
    except Exception as exc:
        message = f"图片分析失败：{exc}"
        return message, "", message


def analyze_streaming(user_input: str):
    global _current_result
    empty = (
        "请先输入要分析的化工异常场景。",
        "分析完成后，这里会展示参考案例、证据链和置信信息。",
        "分析完成后，这里会展示反思日志和物理校验过程。",
        "评估结果将在分析完成后自动生成。",
        "{}",
        None,
        "等待开始分析。",
    )

    if not user_input or not user_input.strip():
        yield empty
        return

    _current_result = {
        "report": None,
        "input": user_input,
        "scores": None,
        "reflection_history": [],
        "image_seed": _current_result.get("image_seed", ""),
    }

    progress_lines = []
    report = None
    context = None
    reflection_history = []

    def push(message: str) -> str:
        progress_lines.append(message)
        return "\n".join(progress_lines)

    progress = push("Step 1/3 上下文解析中：提取设备、参数、偏差方向和工艺条件。")
    yield (
        "分析中，请稍候...",
        "正在整理知识溯源信息...",
        "正在准备反思日志...",
        "评估结果生成中...",
        "{}",
        None,
        progress,
    )

    try:
        ctx_result = ContextAgent().run(user_input.strip())
        context = ctx_result["context"]
        progress = push(
            f"Step 1/3 已完成：{context['equipment']} / "
            f"{context['parameter']} / {context['deviation_direction']}"
        )
    except Exception as exc:
        yield (
            f"上下文解析失败：{exc}",
            "未生成知识溯源。",
            "未生成反思日志。",
            "未生成评估结果。",
            "{}",
            None,
            progress,
        )
        return

    yield (
        "正在执行 RAG 检索与报告生成...",
        "正在汇总参考案例...",
        "等待反思日志...",
        "评估结果生成中...",
        "{}",
        None,
        progress,
    )

    correction_guidance = ""
    round_num = 0

    while round_num < config.MAX_REFLECTION_ROUNDS:
        progress = push(f"Step 2/3 RAG 分析中：第 {round_num + 1} 轮检索和报告生成。")
        yield (
            "正在执行 RAG 检索与报告生成...",
            "正在汇总参考案例...",
            "等待反思日志...",
            "评估结果生成中...",
            "{}",
            None,
            progress,
        )

        try:
            report = RAGAgent().run(
                context=context,
                correction_guidance=correction_guidance,
                reflection_rounds=round_num,
            )
            progress = push("Step 2/3 已完成：已生成结构化 HAZOP 报告草案。")
        except Exception as exc:
            yield (
                f"RAG 报告生成失败：{exc}",
                "未生成知识溯源。",
                "未生成反思日志。",
                "未生成评估结果。",
                "{}",
                None,
                progress,
            )
            return

        progress = push(f"Step 3/3 物理反思中：第 {round_num + 1} 轮双层校验。")
        yield (
            "正在执行物理校验与反思修正...",
            render_sources_panel(report),
            "正在生成反思日志...",
            "评估结果生成中...",
            json.dumps(report, ensure_ascii=False, indent=2),
            None,
            progress,
        )

        try:
            ref_result = ReflectionAgent().run(report)
            record = {
                "round": round_num + 1,
                "passed": ref_result["passed"],
                "issues": ref_result["issues"],
                "fallacy_hits": ref_result.get("fallacy_hits", []),
            }
            reflection_history.append(record)

            if ref_result["passed"]:
                progress = push("Step 3/3 已完成：通过双层物理校验。")
                break

            correction_guidance = ref_result["correction_guidance"]
            progress = push(
                f"Step 3/3 发现 {len(ref_result['issues'])} 个问题，返回第 2 步修正。"
            )
            round_num += 1
        except Exception as exc:
            progress = push(f"物理校验异常，保留当前报告：{exc}")
            break

    if not report:
        yield (
            "分析失败，未生成报告。",
            "未生成知识溯源。",
            "未生成反思日志。",
            "未生成评估结果。",
            "{}",
            None,
            progress,
        )
        return

    report.setdefault("analysis_metadata", {})
    report["analysis_metadata"]["reflection_rounds"] = len(reflection_history) or 1

    report_md = render_report(report)
    sources_md = render_sources_panel(report)
    reflection_md = render_reflection_panel(reflection_history)

    scores = None
    scores_md = "评估结果暂不可用。"
    try:
        scores = evaluate_report(report)
        scores_md = format_scores_markdown(scores)
    except Exception as exc:
        scores_md = f"评估失败：{exc}"

    raw_json = json.dumps(report, ensure_ascii=False, indent=2)
    export_path = None
    try:
        with open(EXPORT_PATH, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        export_path = EXPORT_PATH
    except Exception:
        export_path = None

    _current_result = {
        "report": report,
        "input": user_input,
        "scores": scores,
        "reflection_history": reflection_history,
        "image_seed": _current_result.get("image_seed", ""),
    }

    progress = push("分析完成：可查看 HAZOP 报告、知识溯源、反思日志和原始 JSON。")
    yield (
        report_md,
        sources_md,
        reflection_md,
        scores_md,
        raw_json,
        export_path,
        progress,
    )


def send_image_seed_to_text(seed_text: str):
    if not seed_text or not seed_text.strip():
        return "", "暂无可回填内容，请先完成图片分析。"
    return seed_text, "已将图片识别结果写入文本分析输入框，可切换到“HAZOP 文本分析”继续分析。"


def submit_feedback(rating: int, comment: str):
    if not _current_result.get("report"):
        return "请先完成一次分析再提交反馈。"

    try:
        save_feedback(
            user_input=_current_result["input"],
            report=_current_result["report"],
            rating=rating,
            comment=comment,
            scores=_current_result.get("scores"),
        )
        return f"反馈已记录，评分：{rating}/5。"
    except Exception as exc:
        return f"反馈提交失败：{exc}"


def refresh_stats():
    return format_stats_markdown(get_stats())


def create_ui():
    with gr.Blocks(
        title="DK-SAR Chemical Safety Analysis",
        fill_height=True,
    ) as demo:
        gr.HTML(
            """
            <div style="padding:24px 0 12px 0;text-align:center;">
              <div style="font-size:30px;font-weight:700;color:#0f766e;">DK-SAR 化工安全分析系统</div>
              <div style="margin-top:6px;color:#334155;font-size:15px;">
                Dual Knowledge-enhanced Self-Adaptive Reasoning for Automated HAZOP Analysis
              </div>
              <div style="margin-top:12px;">
                <img src="https://img.shields.io/badge/Gradio-5.x-0f766e?style=flat-square" />
                <img src="https://img.shields.io/badge/Mode-Text%20%2B%20Vision-14b8a6?style=flat-square" />
                <img src="https://img.shields.io/badge/RAG-Positive%20Cases-0ea5e9?style=flat-square" />
                <img src="https://img.shields.io/badge/Reflection-Negative%20Physics%20Checks-f97316?style=flat-square" />
              </div>
            </div>
            """
        )

        with gr.Tabs():
            with gr.Tab("HAZOP 文本分析"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=320):
                        gr.Markdown("### 输入场景")
                        user_input = gr.Textbox(
                            label="化工异常场景描述",
                            placeholder="例如：3号反应釜温度持续升高，压力同步上升，冷却水温偏高，操作员已开启泄压阀。",
                            lines=8,
                        )
                        gr.Examples(
                            examples=EXAMPLES,
                            inputs=[user_input],
                            label="预置示例场景",
                        )
                        analyze_btn = gr.Button("开始 HAZOP 分析", variant="primary", size="lg")

                        with gr.Accordion("使用建议", open=False):
                            gr.Markdown(
                                "\n".join(
                                    [
                                        "- 尽量写清设备、参数、偏差方向和现场操作。",
                                        "- 如果来自拍照识别，可直接使用右侧 Tab 回填后的文本。",
                                        "- 下面这些关键词也可以直接触发快速测试："
                                        f"{' / '.join(TEXT_TAB_EXAMPLES)}",
                                    ]
                                )
                            )

                        with gr.Accordion("用户反馈", open=False):
                            rating_slider = gr.Slider(1, 5, value=4, step=1, label="评分")
                            comment_box = gr.Textbox(
                                label="反馈意见",
                                placeholder="这次分析是否准确、哪里还可以更好。",
                                lines=3,
                            )
                            feedback_btn = gr.Button("提交反馈")
                            feedback_result = gr.Markdown()

                    with gr.Column(scale=2):
                        progress_output = gr.Markdown("等待开始分析。")
                        with gr.Tabs():
                            with gr.Tab("HAZOP 报告"):
                                report_output = gr.Markdown()
                            with gr.Tab("知识溯源"):
                                sources_output = gr.Markdown(
                                    "分析完成后，这里会展示参考案例、证据链和置信信息。"
                                )
                            with gr.Tab("反思日志"):
                                reflection_output = gr.Markdown(
                                    "分析完成后，这里会展示反思日志和物理校验过程。"
                                )
                            with gr.Tab("论文指标评估"):
                                scores_output = gr.Markdown(
                                    "评估结果将在分析完成后自动生成。"
                                )
                            with gr.Tab("原始 JSON"):
                                json_output = gr.Code(language="json", value="{}")

                        export_file = gr.File(label="导出 JSON 报告", interactive=False)
                        stats_output = gr.Markdown(value=format_stats_markdown(get_stats()))
                        refresh_btn = gr.Button("刷新统计", size="sm")

            with gr.Tab("现场拍照识别"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=320):
                        image_input = gr.Image(
                            label="上传现场照片",
                            type="filepath",
                            sources=["upload", "clipboard"],
                            height=360,
                        )
                        image_context = gr.Textbox(
                            label="补充说明",
                            placeholder="例如：这是 3 号车间冷却水管道附近，设备已运行 2 年。",
                            lines=3,
                        )
                        image_btn = gr.Button("开始图片风险识别", variant="primary", size="lg")
                        image_status = gr.Markdown("等待上传照片。")

                    with gr.Column(scale=2):
                        image_output = gr.Markdown(
                            "图片分析完成后，这里会展示风险等级、风险识别、建议措施和 HAZOP 关联。"
                        )
                        image_seed = gr.Textbox(
                            label="转写后的 HAZOP 输入草稿",
                            lines=10,
                            interactive=False,
                        )
                        transfer_status = gr.Markdown("识别完成后可一键回填到文本分析。")
                        transfer_btn = gr.Button("将识别结果转为 HAZOP 分析")

        analyze_btn.click(
            fn=analyze_streaming,
            inputs=[user_input],
            outputs=[
                report_output,
                sources_output,
                reflection_output,
                scores_output,
                json_output,
                export_file,
                progress_output,
            ],
        )
        image_btn.click(
            fn=analyze_image_risk,
            inputs=[image_input, image_context],
            outputs=[image_output, image_seed, image_status],
        )
        transfer_btn.click(
            fn=send_image_seed_to_text,
            inputs=[image_seed],
            outputs=[user_input, transfer_status],
        )
        feedback_btn.click(
            fn=submit_feedback,
            inputs=[rating_slider, comment_box],
            outputs=[feedback_result],
        )
        refresh_btn.click(fn=refresh_stats, outputs=[stats_output])

        gr.HTML(
            """
            <div style="margin-top:20px;padding:18px 0;border-top:1px solid #dbe4e8;color:#475569;font-size:14px;">
              <div><strong>技术架构：</strong>LangGraph 多智能体编排 · 双知识库 RAG/反思校验 · DashScope 多模态视觉分析</div>
              <div style="margin-top:6px;"><strong>论文：</strong>DK-SAR: Dual Knowledge-enhanced Self-Adaptive Reasoning for Automated HAZOP Analysis</div>
              <div style="margin-top:6px;"><strong>作者：</strong>陈仕透，华东理工大学</div>
              <div style="margin-top:6px;"><strong>GitHub：</strong><a href="https://github.com/yilubanhan123c-pixel/dk-sar" target="_blank">yilubanhan123c-pixel/dk-sar</a></div>
            </div>
            """
        )

    return demo


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DK-SAR 系统启动中...")
    print("=" * 60)
    initialize_system()
    create_ui().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=THEME,
    )
