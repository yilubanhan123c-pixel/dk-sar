"""
CSB事故报告批量下载 v2
修复：正确解析CSB网站的案例下拉列表
"""
import os
import json
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path

OUTPUT_DIR = Path("csb_reports")
OUTPUT_DIR.mkdir(exist_ok=True)

BASE_URL = "https://www.csb.gov"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def get_all_cases():
    """从CSB完成调查页面的下拉菜单获取所有案例"""
    print("正在获取CSB案例列表...")
    resp = requests.get(f"{BASE_URL}/investigations/completed-investigations/", headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    
    cases = []
    # 找到下拉菜单中的所有option
    for select in soup.find_all("select"):
        for option in select.find_all("option"):
            name = option.get_text(strip=True)
            value = option.get("value", "")
            if name and name != "-- ALL --" and value:
                cases.append({"name": name, "filter_value": value})
    
    # 如果下拉菜单没找到，尝试从页面链接中提取
    if not cases:
        # 尝试直接抓取所有investigation链接
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)
            if text and len(text) > 10 and ("/investigations/" not in href or href.count("/") > 3):
                # 跳过导航链接
                if any(skip in text.lower() for skip in ["current", "completed", "about", "menu", "search", "home", "recommendations"]):
                    continue
                full_url = href if href.startswith("http") else BASE_URL + href
                cases.append({"name": text, "url": full_url})
    
    print(f"找到 {len(cases)} 个案例")
    return cases

def get_case_page_url(case_name):
    """根据案例名构造详情页URL"""
    # CSB的URL模式：把案例名转成URL slug
    slug = case_name.lower()
    slug = slug.replace("&", "and").replace(",", "").replace(".", "")
    slug = slug.replace("(", "").replace(")", "").replace("/", "-")
    slug = slug.replace("  ", " ").replace(" ", "-")
    slug = slug.strip("-")
    return f"{BASE_URL}/{slug}/"

def find_pdfs_on_page(url):
    """从页面中找所有PDF链接"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        
        pdfs = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True).lower()
            if ".pdf" in href.lower():
                full_url = href if href.startswith("http") else BASE_URL + href
                # 优先级：final report > investigation report > report > 其他
                priority = 0
                if "final" in text or "final" in href.lower():
                    priority = 3
                elif "investigation" in text or "report" in text:
                    priority = 2
                elif "report" in href.lower():
                    priority = 1
                pdfs.append({"url": full_url, "text": a.get_text(strip=True), "priority": priority})
        
        pdfs.sort(key=lambda x: x["priority"], reverse=True)
        return pdfs
    except Exception as e:
        return []

def download_pdf(url, filename):
    """下载PDF"""
    filepath = OUTPUT_DIR / filename
    if filepath.exists():
        print(f"    已存在，跳过")
        return True
    try:
        resp = requests.get(url, headers=HEADERS, timeout=120, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "pdf" not in content_type and "octet" not in content_type:
            return False
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        size_kb = filepath.stat().st_size // 1024
        print(f"    ✅ 下载成功 ({size_kb}KB)")
        return True
    except:
        return False

def try_documents_page():
    """直接从CSB的文档页面获取所有PDF"""
    print("\n尝试从CSB文档页面获取PDF列表...")
    urls_to_try = [
        f"{BASE_URL}/documents/",
        f"{BASE_URL}/investigations/completed-investigations/?Type=2",
    ]
    
    all_pdfs = {}
    for url in urls_to_try:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if ".pdf" in href.lower():
                    full_url = href if href.startswith("http") else BASE_URL + href
                    text = a.get_text(strip=True)
                    if text and len(text) > 3:
                        all_pdfs[full_url] = text
        except:
            continue
    
    print(f"从文档页面找到 {len(all_pdfs)} 个PDF")
    return all_pdfs

def main():
    print("=" * 60)
    print("  CSB事故报告批量下载 v2")
    print("=" * 60)
    
    # 方法1：从完成调查页面获取案例列表，逐个访问详情页找PDF
    cases = get_all_cases()
    
    # 预定义的重要案例URL（确保这些一定能下载到）
    known_cases = {
        "BP Texas City": f"{BASE_URL}/bp-america-texas-city-refinery-explosion/",
        "T2 Laboratories": f"{BASE_URL}/t2-laboratories-inc-reactive-chemical-explosion/",
        "Imperial Sugar": f"{BASE_URL}/imperial-sugar-company-dust-explosion-and-fire/",
        "Chevron Richmond": f"{BASE_URL}/chevron-richmond-refinery-fire/",
        "Tesoro Anacortes": f"{BASE_URL}/tesoro-anacortes-refinery-fatal-explosion-and-fire/",
        "West Fertilizer": f"{BASE_URL}/west-fertilizer-explosion-and-fire/",
        "Caribbean Petroleum": f"{BASE_URL}/caribbean-petroleum-corporation-capeco-refinery-tank-explosion-and-fire/",
        "DuPont La Porte": f"{BASE_URL}/dupont-la-porte-facility-toxic-chemical-release/",
        "Williams Olefins": f"{BASE_URL}/williams-olefins-plant-explosion-and-fire/",
        "Philadelphia Energy": f"{BASE_URL}/philadelphia-energy-solutions-pes-refinery-fire-and-explosions/",
        "Husky Energy": f"{BASE_URL}/husky-energy-superior-refinery-explosion-and-fire/",
        "Morton International": f"{BASE_URL}/morton-international-inc-runaway-chemical-reaction/",
        "Nitrogen Asphyxiation": f"{BASE_URL}/hazards-of-nitrogen-asphyxiation/",
        "Formosa Plastics VCM": f"{BASE_URL}/formosa-plastics-vinyl-chloride-explosion/",
        "Formosa Plastics Propylene": f"{BASE_URL}/formosa-plastics-propylene-explosion/",
        "Bayer CropScience": f"{BASE_URL}/bayer-cropscience-pesticide-waste-tank-explosion/",
        "Synthron": f"{BASE_URL}/synthron-chemical-explosion/",
        "KMCO": f"{BASE_URL}/kmco-llc-fatal-fire-and-explosion/",
        "ITC Tank Fire": f"{BASE_URL}/intercontinental-terminals-company-itc-tank-fire/",
        "ExxonMobil Torrance": f"{BASE_URL}/exxonmobil-torrance-refinery-explosion/",
        "TPC Port Neches": f"{BASE_URL}/tpc-port-neches-explosions-and-fire/",
        "Valero McKee": f"{BASE_URL}/valero-mckee-refinery-propane-fire/",
        "DPC Chlorine": f"{BASE_URL}/dpc-enterprises-festus-chlorine-release/",
        "Motiva Sulfuric Acid": f"{BASE_URL}/motiva-enterprises-sulfuric-acid-tank-explosion/",
        "Giant Industries": f"{BASE_URL}/giant-industries-refinery-explosions-and-fire/",
        "Concept Sciences": f"{BASE_URL}/concept-sciences-hydroxylamine-explosion/",
        "Barton Solvents": f"{BASE_URL}/barton-solvents-explosions-and-fire/",
        "Hoeganaes Flash Fires": f"{BASE_URL}/hoeganaes-corporation-fatal-flash-fires/",
        "DuPont Hotwork": f"{BASE_URL}/e-i-dupont-de-nemours-co-fatal-hotwork-explosion/",
        "Kleen Energy": f"{BASE_URL}/kleen-energy-natural-gas-explosion/",
        "Silver Eagle": f"{BASE_URL}/silver-eagle-refinery-flash-fire-and-explosion-and-catastrophic-pipe-explosion/",
        "Tosco Avon": f"{BASE_URL}/tosco-avon-refinery-petroleum-naphtha-fire/",
        "First Chemical": f"{BASE_URL}/first-chemical-corp-reactive-chemical-explosion/",
        "Georgia Pacific H2S": f"{BASE_URL}/georgia-pacific-corp-hydrogen-sulfide-poisoning/",
        "Goodyear Heat Exchanger": f"{BASE_URL}/goodyear-heat-exchanger-rupture/",
        "Carbide Industries": f"{BASE_URL}/carbide-industries-fire-and-explosion/",
        "Union Carbide N2": f"{BASE_URL}/union-carbide-corp-nitrogen-asphyxiation-incident/",
        "Valero Asphyxiation": f"{BASE_URL}/valero-delaware-city-refinery-asphyxiation-incident/",
        "Dust Hazard Study": f"{BASE_URL}/combustible-dust-hazard-investigation/",
        "Reactive Hazard": f"{BASE_URL}/improving-reactive-hazard-management/",
    }
    
    download_count = 0
    failed = []
    
    print(f"\n开始下载 {len(known_cases)} 个重点案例的PDF...\n")
    
    for i, (name, url) in enumerate(known_cases.items()):
        print(f"[{i+1}/{len(known_cases)}] {name}")
        print(f"  URL: {url}")
        
        pdfs = find_pdfs_on_page(url)
        
        if pdfs:
            best = pdfs[0]
            safe_name = name.replace("/", "-").replace(" ", "_")[:60]
            filename = f"{safe_name}.pdf"
            
            if download_pdf(best["url"], filename):
                download_count += 1
            else:
                print(f"    ❌ 下载失败")
                failed.append(name)
        else:
            print(f"    ⚠️ 未找到PDF，尝试搜索assets...")
            # 尝试直接访问常见的PDF路径模式
            alt_tried = False
            for pattern in ["/assets/1/20/", "/assets/1/6/", "/file.aspx?DocumentId="]:
                try:
                    resp = requests.get(url, headers=HEADERS, timeout=15)
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for a in soup.find_all("a", href=True):
                        if ".pdf" in a["href"].lower() or "file.aspx" in a["href"].lower():
                            pdf_url = a["href"] if a["href"].startswith("http") else BASE_URL + a["href"]
                            safe_name = name.replace("/", "-").replace(" ", "_")[:60]
                            if download_pdf(pdf_url, f"{safe_name}.pdf"):
                                download_count += 1
                                alt_tried = True
                                break
                    if alt_tried:
                        break
                except:
                    continue
            
            if not alt_tried:
                failed.append(name)
                print(f"    ❌ 未找到可下载的PDF")
        
        time.sleep(1.5)
    
    # 保存结果
    log = {
        "total_attempted": len(known_cases),
        "downloaded": download_count,
        "failed": failed,
        "output_dir": str(OUTPUT_DIR)
    }
    with open(OUTPUT_DIR / "download_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"下载完成！")
    print(f"成功: {download_count} 个")
    print(f"失败: {len(failed)} 个")
    if failed:
        print(f"失败列表: {', '.join(failed[:10])}...")
    print(f"PDF保存在: {OUTPUT_DIR.absolute()}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
