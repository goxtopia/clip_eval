import os
import re
import sys
import json
from collections import OrderedDict

from bs4 import BeautifulSoup


def normalize_label(s: str) -> str:
    """统一 key：去掉多余空格 + 转小写。"""
    return " ".join(s.strip().lower().split())


def parse_report(html_path: str):
    """
    解析单个 i2t 或 t2i 报告：
    返回 (global_metrics, per_class_dict)
    global_metrics: dict, 包含 micro_top1 / micro_top5 / macro_top1 / macro_top5
    per_class_dict: {normalized_label: {label, count, top1, top5}}
    """
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # 1. 解析 Global / Macro 指标
    global_metrics = {}
    footers = soup.select(".card-footer")

    # 约定：第一个 card-footer 是 Global(Micro)，第二个是 Macro
    kinds = ["micro", "macro"]
    for idx, kind in enumerate(kinds):
        if idx >= len(footers):
            break
        text = footers[idx].get_text(strip=True)
        m1 = re.search(r"Top-1:\s*([\d.]+)%", text)
        m5 = re.search(r"Top-5:\s*([\d.]+)%", text)
        if m1:
            global_metrics[f"{kind}_top1"] = float(m1.group(1))
        if m5:
            global_metrics[f"{kind}_top5"] = float(m5.group(1))

    # 2. 解析 per-class 表格
    per_class = {}
    tbody = soup.find("tbody", id="tableBody")
    if not tbody:
        return global_metrics, per_class

    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue
        label = tds[0].get_text(strip=True)
        count_str = tds[1].get_text(strip=True)
        top1_str = tds[2].get_text(strip=True)
        top5_str = tds[3].get_text(strip=True)

        try:
            count = int(count_str)
        except ValueError:
            continue

        def parse_pct(s: str) -> float:
            return float(s.replace("%", "").strip())

        try:
            top1 = parse_pct(top1_str)
            top5 = parse_pct(top5_str)
        except ValueError:
            continue

        key = normalize_label(label)
        per_class[key] = {
            "label": label,
            "count": count,
            "top1": top1,
            "top5": top5,
        }

    return global_metrics, per_class


def find_model_reports(root_dir: str):
    """
    在 root_dir 下寻找每个模型文件夹，并解析其中 i2t / t2i 报告。
    返回：
    models = {
        model_name: {
            "i2t_global": {...},
            "t2i_global": {...},
            "i2t_per_class": {...},
            "t2i_per_class": {...},
        }
    }
    """
    models = OrderedDict()

    for name in sorted(os.listdir(root_dir)):
        model_dir = os.path.join(root_dir, name)
        if not os.path.isdir(model_dir):
            continue

        i2t_path = None
        t2i_path = None

        for fn in os.listdir(model_dir):
            lfn = fn.lower()
            if "i2t" in lfn and lfn.endswith(".html"):
                i2t_path = os.path.join(model_dir, fn)
            elif "t2i" in lfn and lfn.endswith(".html"):
                t2i_path = os.path.join(model_dir, fn)

        # 至少有一个就记录下来（你也可以改成必须 i2t+t2i 成对）
        if not i2t_path and not t2i_path:
            continue

        model_info = {
            "i2t_global": {},
            "t2i_global": {},
            "i2t_per_class": {},
            "t2i_per_class": {},
        }

        if i2t_path:
            gi, pc_i = parse_report(i2t_path)
            model_info["i2t_global"] = gi
            model_info["i2t_per_class"] = pc_i

        if t2i_path:
            gt, pc_t = parse_report(t2i_path)
            model_info["t2i_global"] = gt
            model_info["t2i_per_class"] = pc_t

        models[name] = model_info

    return models


def choose_top_classes(models, top_k=10):
    """
    用第一个模型的 i2t per-class，根据 count 排序，选前 top_k 个类别。
    返回：(top_keys, key_to_label, key_to_count)
    """
    if not models:
        return [], {}, {}

    first_model = next(iter(models.keys()))
    per_class = models[first_model]["i2t_per_class"]

    sorted_classes = sorted(
        per_class.values(),
        key=lambda x: x["count"],
        reverse=True,
    )
    top_classes = sorted_classes[:top_k]

    top_keys = []
    key_to_label = {}
    key_to_count = {}

    for c in top_classes:
        key = normalize_label(c["label"])
        top_keys.append(key)
        key_to_label[key] = c["label"]
        key_to_count[key] = c["count"]

    return top_keys, key_to_label, key_to_count


def fmt_pct(x):
    return f"{x:.2f}%" if x is not None else "–"


def generate_html(models, out_path: str, top_k: int = 10):
    if not models:
        print("没有找到任何模型报告。")
        return

    model_names = list(models.keys())
    top_keys, key_to_label, key_to_count = choose_top_classes(models, top_k=top_k)

    # -------- 准备图表用数据 --------
    # Global charts: i2t / t2i 的 micro/macro top1/5
    i2t_global = {
        "microTop1": [],
        "microTop5": [],
        "macroTop1": [],
        "macroTop5": [],
    }
    t2i_global = {
        "microTop1": [],
        "microTop5": [],
        "macroTop1": [],
        "macroTop5": [],
    }

    for m in model_names:
        gi = models[m].get("i2t_global", {}) or {}
        gt = models[m].get("t2i_global", {}) or {}

        i2t_global["microTop1"].append(gi.get("micro_top1"))
        i2t_global["microTop5"].append(gi.get("micro_top5"))
        i2t_global["macroTop1"].append(gi.get("macro_top1"))
        i2t_global["macroTop5"].append(gi.get("macro_top5"))

        t2i_global["microTop1"].append(gt.get("micro_top1"))
        t2i_global["microTop5"].append(gt.get("micro_top5"))
        t2i_global["macroTop1"].append(gt.get("macro_top1"))
        t2i_global["macroTop5"].append(gt.get("macro_top5"))

    # Top-class charts: shape = [num_models][num_classes]
    i2t_top1 = []
    i2t_top5 = []
    t2i_top1 = []
    t2i_top5 = []

    for m in model_names:
        pc_i = models[m]["i2t_per_class"]
        pc_t = models[m]["t2i_per_class"]

        arr_i2t_t1 = []
        arr_i2t_t5 = []
        arr_t2i_t1 = []
        arr_t2i_t5 = []

        for key in top_keys:
            r_i = pc_i.get(key)
            r_t = pc_t.get(key)

            arr_i2t_t1.append(r_i["top1"] if r_i else None)
            arr_i2t_t5.append(r_i["top5"] if r_i else None)
            arr_t2i_t1.append(r_t["top1"] if r_t else None)
            arr_t2i_t5.append(r_t["top5"] if r_t else None)

        i2t_top1.append(arr_i2t_t1)
        i2t_top5.append(arr_i2t_t5)
        t2i_top1.append(arr_t2i_t1)
        t2i_top5.append(arr_t2i_t5)

    top_labels = [key_to_label.get(k, k) for k in top_keys]

    # -------- 开始拼 HTML --------
    html = []
    add = html.append

    add("<!DOCTYPE html>")
    add("<html lang='en'>")
    add("<head>")
    add("  <meta charset='UTF-8'>")
    add("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
    add("  <title>CLIP Models Comparison Report</title>")
    add("  <link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css' rel='stylesheet'/>")
    add("  <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>")
    add("  <style>")
    add("    body { padding: 20px; font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }")
    add("    h1, h2, h3 { margin-top: 24px; }")
    add("    table { margin-top: 12px; }")
    add("    .model-table th, .model-table td { white-space: nowrap; }")
    add("    canvas { background-color: #ffffff; }")
    add("  </style>")
    add("</head>")
    add("<body>")
    add("  <div class='container-fluid'>")
    add("    <h1>CLIP Models Comparison Report</h1>")
    add("    <p>汇总每个模型的 Image-to-Text (i2t) 与 Text-to-Image (t2i) 能力对比，并生成全局与小类的柱状图。</p>")

    # ========== 图表区域 ==========
    add("    <h2>Global Performance Charts</h2>")
    add("    <div class='row mb-4'>")
    add("      <div class='col-md-6 mb-3'>")
    add("        <div class='card'>")
    add("          <div class='card-header'>i2t Global Accuracy</div>")
    add("          <div class='card-body'><canvas id='i2tGlobalChart' height='200'></canvas></div>")
    add("        </div>")
    add("      </div>")
    add("      <div class='col-md-6 mb-3'>")
    add("        <div class='card'>")
    add("          <div class='card-header'>t2i Global Accuracy</div>")
    add("          <div class='card-body'><canvas id='t2iGlobalChart' height='200'></canvas></div>")
    add("        </div>")
    add("      </div>")
    add("    </div>")

    add("    <h2>Per-Class Performance Charts (Top-{} by count)</h2>".format(len(top_keys)))
    add("    <p>按第一个模型 i2t 报告中 <strong>Image Count</strong> 最大的前 {} 个类别，分别比较各模型的 Top-1 / Top-5。".format(len(top_keys)))

    add("    <div class='row mb-4'>")
    add("      <div class='col-md-6 mb-3'>")
    add("        <div class='card'>")
    add("          <div class='card-header'>i2t Top-1 Accuracy (Top Classes)</div>")
    add("          <div class='card-body'><canvas id='i2tTop1Chart' height='220'></canvas></div>")
    add("        </div>")
    add("      </div>")
    add("      <div class='col-md-6 mb-3'>")
    add("        <div class='card'>")
    add("          <div class='card-header'>i2t Top-5 Accuracy (Top Classes)</div>")
    add("          <div class='card-body'><canvas id='i2tTop5Chart' height='220'></canvas></div>")
    add("        </div>")
    add("      </div>")
    add("    </div>")

    add("    <div class='row mb-4'>")
    add("      <div class='col-md-6 mb-3'>")
    add("        <div class='card'>")
    add("          <div class='card-header'>t2i Top-1 Accuracy (Top Classes)</div>")
    add("          <div class='card-body'><canvas id='t2iTop1Chart' height='220'></canvas></div>")
    add("        </div>")
    add("      </div>")
    add("      <div class='col-md-6 mb-3'>")
    add("        <div class='card'>")
    add("          <div class='card-header'>t2i Top-5 Accuracy (Top Classes)</div>")
    add("          <div class='card-body'><canvas id='t2iTop5Chart' height='220'></canvas></div>")
    add("        </div>")
    add("      </div>")
    add("    </div>")

    # ========== 表格区域（和之前类似） ==========
    # 1. 全局指标对比表
    add("    <h2>Global Performance Comparison (Table)</h2>")
    add("    <div class='table-responsive'>")
    add("    <table class='table table-striped table-bordered align-middle model-table'>")
    add("      <thead class='table-dark'>")
    add("        <tr>")
    add("          <th rowspan='2'>Model</th>")
    add("          <th colspan='4' class='text-center'>Image-to-Text (i2t)</th>")
    add("          <th colspan='4' class='text-center'>Text-to-Image (t2i)</th>")
    add("        </tr>")
    add("        <tr>")
    add("          <th>Micro Top-1</th><th>Micro Top-5</th>")
    add("          <th>Macro Top-1</th><th>Macro Top-5</th>")
    add("          <th>Micro Top-1</th><th>Micro Top-5</th>")
    add("          <th>Macro Top-1</th><th>Macro Top-5</th>")
    add("        </tr>")
    add("      </thead>")
    add("      <tbody>")

    for m in model_names:
        info = models[m]
        gi = info.get("i2t_global", {})
        gt = info.get("t2i_global", {})
        add("        <tr>")
        add(f"          <td>{m}</td>")
        add(f"          <td>{fmt_pct(gi.get('micro_top1')) if gi else '–'}</td>")
        add(f"          <td>{fmt_pct(gi.get('micro_top5')) if gi else '–'}</td>")
        add(f"          <td>{fmt_pct(gi.get('macro_top1')) if gi else '–'}</td>")
        add(f"          <td>{fmt_pct(gi.get('macro_top5')) if gi else '–'}</td>")
        add(f"          <td>{fmt_pct(gt.get('micro_top1')) if gt else '–'}</td>")
        add(f"          <td>{fmt_pct(gt.get('micro_top5')) if gt else '–'}</td>")
        add(f"          <td>{fmt_pct(gt.get('macro_top1')) if gt else '–'}</td>")
        add(f"          <td>{fmt_pct(gt.get('macro_top5')) if gt else '–'}</td>")
        add("        </tr>")

    add("      </tbody>")
    add("    </table>")
    add("    </div>")

    # 2. 小类别能力对比（i2t）
    add("    <h2>Per-Class Comparison (Top-{} by count, Table)</h2>".format(len(top_keys)))
    add("    <p>按第一个模型 i2t 报告中 Image Count 最大的前 {} 个类别进行比较。</p>".format(len(top_keys)))

    # 2.1 i2t
    add("    <h3>Image-to-Text (i2t) – Top Classes</h3>")
    add("    <div class='table-responsive'>")
    add("    <table class='table table-striped table-bordered align-middle'>")
    add("      <thead class='table-dark'>")
    add("        <tr>")
    add("          <th>Category (Representative Label)</th>")
    add("          <th>Image Count (reference)</th>")
    for m in model_names:
        add(f"          <th>{m}<br><small>Top-1 / Top-5</small></th>")
    add("        </tr>")
    add("      </thead>")
    add("      <tbody>")

    for key in top_keys:
        label = key_to_label.get(key, key)
        count = key_to_count.get(key, "")
        add("        <tr>")
        add(f"          <td>{label}</td>")
        add(f"          <td>{count}</td>")
        for m in model_names:
            per_class_i2t = models[m]["i2t_per_class"]
            row = per_class_i2t.get(key)
            if row:
                t1 = fmt_pct(row['top1'])
                t5 = fmt_pct(row['top5'])
                cell = f"{t1} / {t5}"
            else:
                cell = "–"
            add(f"          <td>{cell}</td>")
        add("        </tr>")

    add("      </tbody>")
    add("    </table>")
    add("    </div>")

    # 2.2 t2i
    add("    <h3>Text-to-Image (t2i) – Top Classes (matched by normalized text)</h3>")
    add("    <div class='table-responsive'>")
    add("    <table class='table table-striped table-bordered align-middle'>")
    add("      <thead class='table-dark'>")
    add("        <tr>")
    add("          <th>Category (using i2t label as reference)</th>")
    add("          <th>Query Count (from matched t2i entries where available)</th>")
    for m in model_names:
        add(f"          <th>{m}<br><small>Top-1 / Top-5</small></th>")
    add("        </tr>")
    add("      </thead>")
    add("      <tbody>")

    for key in top_keys:
        label = key_to_label.get(key, key)
        # 参考 t2i count：取第一个有的模型
        t2i_count = ""
        for m in model_names:
            row = models[m]["t2i_per_class"].get(key)
            if row:
                t2i_count = row["count"]
                break

        add("        <tr>")
        add(f"          <td>{label}</td>")
        add(f"          <td>{t2i_count}</td>")
        for m in model_names:
            per_class_t2i = models[m]["t2i_per_class"]
            row = per_class_t2i.get(key)
            if row:
                t1 = fmt_pct(row['top1'])
                t5 = fmt_pct(row['top5'])
                cell = f"{t1} / {t5}"
            else:
                cell = "–"
            add(f"          <td>{cell}</td>")
        add("        </tr>")

    add("      </tbody>")
    add("    </table>")
    add("    </div>")

    add("    <hr>")
    add("    <p class='text-muted'>Generated by compare_clip_reports.py</p>")
    add("  </div>")  # container

    # ========== 注入 Chart.js 脚本，风格和你现有 HTML 一致 ==========
    add("  <script>")
    add("    const modelNames = " + json.dumps(model_names) + ";")
    add("    const topLabels = " + json.dumps(top_labels) + ";")
    add("    const i2tGlobal = " + json.dumps(i2t_global) + ";")
    add("    const t2iGlobal = " + json.dumps(t2i_global) + ";")
    add("    const i2tTop1 = " + json.dumps(i2t_top1) + ";")
    add("    const i2tTop5 = " + json.dumps(i2t_top5) + ";")
    add("    const t2iTop1 = " + json.dumps(t2i_top1) + ";")
    add("    const t2iTop5 = " + json.dumps(t2i_top5) + ";")

    # 和你原始 HTML 相似的颜色风格
    add("    const colors = [")
    add("      'rgba(54, 162, 235, 0.5)',")
    add("      'rgba(255, 99, 132, 0.5)',")
    add("      'rgba(75, 192, 192, 0.5)',")
    add("      'rgba(255, 206, 86, 0.5)',")
    add("      'rgba(153, 102, 255, 0.5)',")
    add("      'rgba(255, 159, 64, 0.5)'")
    add("    ];")
    add("    const borderColors = [")
    add("      'rgba(54, 162, 235, 1)',")
    add("      'rgba(255, 99, 132, 1)',")
    add("      'rgba(75, 192, 192, 1)',")
    add("      'rgba(255, 206, 86, 1)',")
    add("      'rgba(153, 102, 255, 1)',")
    add("      'rgba(255, 159, 64, 1)'")
    add("    ];")

    add("    function buildGlobalData(data) {")
    add("      const metricKeys = ['microTop1', 'microTop5', 'macroTop1', 'macroTop5'];")
    add("      const labels = ['Micro Top-1', 'Micro Top-5', 'Macro Top-1', 'Macro Top-5'];")
    add("      const datasets = modelNames.map((name, idx) => ({")
    add("        label: name,")
    add("        data: metricKeys.map(k => data[k][idx] ?? null),")
    add("        backgroundColor: colors[idx % colors.length],")
    add("        borderColor: borderColors[idx % borderColors.length],")
    add("        borderWidth: 1")
    add("      }));")
    add("      return { labels, datasets };")
    add("    }")

    add("    function buildTopData(values) {")
    add("      // values: [numModels][numClasses]")
    add("      const datasets = modelNames.map((name, idx) => ({")
    add("        label: name,")
    add("        data: values[idx],")
    add("        backgroundColor: colors[idx % colors.length],")
    add("        borderColor: borderColors[idx % borderColors.length],")
    add("        borderWidth: 1")
    add("      }));")
    add("      return { labels: topLabels, datasets };")
    add("    }")

    add("    function makeBarChart(canvasId, chartData, titleText) {")
    add("      const ctx = document.getElementById(canvasId).getContext('2d');")
    add("      new Chart(ctx, {")
    add("        type: 'bar',")
    add("        data: chartData,")
    add("        options: {")
    add("          responsive: true,")
    add("          plugins: {")
    add("            legend: { position: 'top' },")
    add("            title: { display: !!titleText, text: titleText }")
    add("          },")
    add("          scales: {")
    add("            y: { beginAtZero: true, max: 100, title: { display: true, text: 'Accuracy (%)' } },")
    add("            x: { ticks: { maxRotation: 60, minRotation: 30 } }")
    add("          }")
    add("        }")
    add("      });")
    add("    }")

    add("    // 生成图表（加载完成后）")
    add("    window.addEventListener('load', function () {")
    add("      makeBarChart('i2tGlobalChart', buildGlobalData(i2tGlobal), 'i2t Global Accuracy');")
    add("      makeBarChart('t2iGlobalChart', buildGlobalData(t2iGlobal), 't2i Global Accuracy');")
    add("      makeBarChart('i2tTop1Chart', buildTopData(i2tTop1), 'i2t Top-1 on Top Classes');")
    add("      makeBarChart('i2tTop5Chart', buildTopData(i2tTop5), 'i2t Top-5 on Top Classes');")
    add("      makeBarChart('t2iTop1Chart', buildTopData(t2iTop1), 't2i Top-1 on Top Classes');")
    add("      makeBarChart('t2iTop5Chart', buildTopData(t2iTop5), 't2i Top-5 on Top Classes');")
    add("    });")

    add("  </script>")
    add("</body>")
    add("</html>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"对比报告已保存到: {out_path}")


def main():
    # 用法：python compare_clip_reports.py [root_dir] [out_html]
    if len(sys.argv) >= 2:
        root_dir = sys.argv[1]
    else:
        root_dir = "."

    if len(sys.argv) >= 3:
        out_html = sys.argv[2]
    else:
        out_html = "comparison_report.html"

    models = find_model_reports(root_dir)
    if not models:
        print("在指定目录下没有找到任何包含 i2t/t2i 报告的模型文件夹。")
        return

    generate_html(models, out_html, top_k=10)


if __name__ == "__main__":
    main()

