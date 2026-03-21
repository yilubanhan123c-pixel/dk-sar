"""补充正样本库到120条 - 用AI直接生成"""
import os, json, time
from dashscope import Generation

API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 还没收录的CSB案例 + 其他经典化工事故
NEW_CASES = [
    # 之前下载PDF失败的9个
    "Tesoro Anacortes Refinery Fatal Explosion 2010",
    "West Fertilizer Explosion and Fire 2013",
    "DuPont La Porte Facility Toxic Chemical Release 2014",
    "Williams Olefins Plant Explosion and Fire 2013",
    "Philadelphia Energy Solutions Refinery Fire 2019",
    "KMCO LLC Fatal Fire and Explosion 2019",
    "ExxonMobil Torrance Refinery Explosion 2015",
    "First Chemical Corp Reactive Chemical Explosion 2002",
    "Carbide Industries Fire and Explosion 2003",
    # 补充更多经典案例
    "Arkema Inc Chemical Plant Fire Hurricane Harvey 2017",
    "Deepwater Horizon Macondo Blowout 2010",
    "ConAgra Natural Gas Explosion 2009",
    "Didion Milling Dust Explosion 2017",
    "Honeywell Baton Rouge Chemical Release 2003",
    "Foundation Food Group Liquid Nitrogen Release 2021",
    "Praxair Flammable Gas Cylinder Fire 2005",
    "Wacker Polysilicon Chemical Release 2021",
    "Kuraray Pasadena Chemical Release 2018",
    "NDK Crystal Explosion with Offsite Fatality 2009",
    "Horsehead Holding Fatal Explosion 2010",
    "LyondellBasell La Porte Fatal Chemical Release 2014",
    "PEMEX Deer Park Chemical Release 2016",
    "Optima Belle Explosion and Fire 2016",
    "Watson Grinding Fatal Explosion 2020",
    "Midland Resource Recovery Explosion 2020",
    "Shell Norco Refinery Explosion 1988",
    "Phillips 66 Pasadena Polyethylene Plant Explosion 1989",
    "Flixborough Disaster UK Cyclohexane Explosion 1974",
    "Piper Alpha North Sea Platform Explosion 1988",
    "Bhopal Union Carbide Methyl Isocyanate Release 1984",
    "Toulouse AZF Ammonium Nitrate Explosion 2001",
    "Texas City BP Amoco Refinery BLEVE 1985",
    "Longford Esso Gas Plant Explosion Australia 1998",
    "Buncefield Oil Storage Depot Explosion UK 2005",
    "Tianjin Port Chemical Explosion China 2015",
    "Jiangsu Xiangshui Chemical Plant Explosion China 2019",
    "Beirut Port Ammonium Nitrate Explosion 2020",
    "Marathon Petroleum Refinery HF Release Texas City 1987",
    "Hickson Welch Chemical Reactor Explosion UK 1992",
]

def generate_case(name, idx):
    prompt = f"""你是化工安全HAZOP分析专家。请基于以下真实化工事故的公开信息，生成结构化案例JSON。

事故：{name}

输出严格JSON格式（不要```标记），字段如下：
{{
  "case_id": "P{idx:03d}",
  "name": "事故英文名称",
  "year": "年份",
  "fatalities": 数字,
  "process_type": "工艺类型",
  "equipment": ["关键设备"],
  "core_scenario": "事故核心场景描述（中文，50字内）",
  "deviation": "HAZOP偏差类型",
  "causes": ["根本原因1", "根本原因2"],
  "consequences": ["后果1", "后果2"],
  "safeguards": ["已有保护措施"],
  "recommendations": ["改进建议1", "改进建议2"],
  "key_physics": "涉及的物理原理",
  "embedding_text": "100词英文概括事故核心机理，含设备、物料、偏差、根因、后果"
}}"""

    try:
        resp = Generation.call(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            api_key=API_KEY, result_format="message"
        )
        r = resp.output.choices[0].message.content.strip()
        if r.startswith("```json"): r = r[7:]
        if r.startswith("```"): r = r[3:]
        if r.endswith("```"): r = r[:-3]
        return json.loads(r.strip())
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        return None

def main():
    if not API_KEY:
        k = input("请输入DASHSCOPE_API_KEY: ").strip()
        os.environ["DASHSCOPE_API_KEY"] = k

    # 读取现有案例
    with open("data/positive/cases.json", "r", encoding="utf-8") as f:
        existing = json.load(f)
    
    current_count = len(existing)
    need = 120 - current_count
    print(f"当前 {current_count} 条，需要补充 {need} 条到120条\n")
    
    if need <= 0:
        print("已经够120条了！")
        return
    
    cases_to_add = NEW_CASES[:need]
    added = []
    
    for i, name in enumerate(cases_to_add):
        idx = current_count + i + 1
        print(f"[{i+1}/{len(cases_to_add)}] {name}")
        case = generate_case(name, idx)
        if case:
            case["case_id"] = f"P{idx:03d}"
            added.append(case)
            print(f"  ✅ 成功")
        time.sleep(2)
    
    # 合并
    merged = existing + added
    for i, c in enumerate(merged):
        c["case_id"] = f"P{i+1:03d}"
    
    with open("data/positive/cases.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    print(f"\n完成！{current_count} + {len(added)} = {len(merged)} 条")
    print("请删除 data/chroma_db 后重启 app.py 重建索引")

if __name__ == "__main__":
    main()
