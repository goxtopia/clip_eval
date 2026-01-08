import os
import re
import sys
import json
import statistics
from collections import OrderedDict

from bs4 import BeautifulSoup


# 默认的人相关关键词（可在命令行参数中覆盖）
DEFAULT_KEYWORDS = [
    "person",
    "people",
    "man",
    "woman",
    "men",
    "women",
    "boy",
    "girl",
    "human",
]


def normalize_label(s: str) -> str:
    """统一 label / 文本的 key：去掉多余空格并转小写。"""
    return " ".join(s.strip().lower().split())


def parse_i2t_per_class(html_path: str):
    """
    解析 i2t 报告，只取 per-class 信息：
    返回 dict: {normalized_label: {label, count}}
    其中 count 是 Image Count（真实图片数量）
    """
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    tbody = soup.find("tbody", id="tableBody")
    if not tbody:
        return {}

    per_class = {}
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue

        label = tds[0].get_text(strip=True)
        count_str = tds[1].get_text(strip=True)

        try:
            count = int(count_str)
        except ValueError:
            continue

        key = normalize_label(label)
        per_class[key] = {
            "label": label,
            "count": count,
        }

    return per_class


def parse_t2i_people_rows(html_path: str, keywords=None):
    """
    解析 t2i 报告中与“人”相关的查询行：
    - Query Text 中只要包含任一关键词就算
    返回列表：
      rows = [
        {
          "text": query_text,
          "key": normalized_key,
          "query_count": count,
          "top1": top1,
          "top5": top5,
        },
        ...
      ]
    """
    if keywords is None:
        keywords = DEFAULT_KEYWORDS

    kw_lower = [k.lower() for k in keywords]
    # 用单词边界避免 environment 中误匹配 men
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, kw_lower)) + r")\b")

    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    tbody = soup.find("tbody", id="tableBody")
    if not tbody:
        return []

    rows = []
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue

        query_text = tds[0].get_text(strip=True)
        if not pattern.search(query_text.lower()):
            continue  # 不含任何人相关关键词

        count_str = tds[1].get_text(strip=True)
        top1_str = tds[2].get_text(strip=True)
        top5_str = tds[3].get_text(strip=True)

        try:
            query_count = int(count_str)
        except ValueError:
            continue

        def parse_pct(s: str) -> float:
            return float(s.replace("%", "").strip())

        try:
            top1 = parse_pct(top1_str)
            top5 = parse_pct(top5_str)
        except ValueError:
            continue

        key = normalize_label(query_text)
        rows.append(
            {
                "text": query_text,
                "key": key,
                "query_count": query_count,
                "top1": top1,
                "top5": top5,
            }
        )

    return rows


def aggregate_people_with_i2t_weights(rows, i2t_per_class):
    """
    rows: parse_t2i_people_rows 结果
    i2t_per_class: parse_i2t_per_class 结果

    输出：
    {
        "num_queries":  匹配到的人相关 query 条数（只要 t2i 里有就算）
        "total_t2i_queries": t2i 中这些 query 的 Query Count 总和（纯信息）
        "macro_top1":   等权平均 Top-1
        "macro_top5":   等权平均 Top-5
        "weighted_top1": 使用 i2t Image Count 作为权重的加权 Top-1（可能为 None）
        "weighted_top5": 同上
        "weighted_image_count": 参与加权的 i2t 图片总数（权重之和）
    """
    if not rows:
        return None

    # 等权综合
    top1s = [r["top1"] for r in rows]
    top5s = [r["top5"] for r in rows]

    macro_top1 = statistics.mean(top1s)
    macro_top5 = statistics.mean(top5s)

    total_t2i_queries = sum(r["query_count"] for r in rows)

    # 使用 i2t 的 Image Count 做权重
    num_queries = len(rows)
    weighted_numerator1 = 0.0
    weighted_numerator5 = 0.0
    weighted_denom = 0

    for r in rows:
        key = r["key"]
        i2t_row = i2t_per_class.get(key)
        if not i2t_row:
            # 在 i2t 里找不到对应类别，就不参加加权
            continue
        w = i2t_row["count"]
        if w <= 0:
            continue

        weighted_denom += w
        weighted_numerator1 += r["top1"] * w
        weighted_numerator5 += r["top5"] * w

    if weighted_denom > 0:
        weighted_top1 = weighted_numerator1 / weighted_denom
        weighted_top5 = weighted_numerator5 / weighted_denom
    else:
        weighted_top1 = None
        weighted_top5 = None

    return {
        "num_queries": num_queries,
        "total_t2i_queries": int(total_t2i_queries),
        "macro_top1": macro_top1,
        "macro_top5": macro_top5,
        "weighted_top1": weighted_top1,
        "weighted_top5": weighted_top5,
        "weighted_image_count": weighted_denom,
    }


def find_model_reports(root_dir: str):
    """
    在 root_dir 下寻找每个模型文件夹，找到其中 i2t / t2i 报告：
    - i2t: 文件名包含 'i2t' 且以 .html 结尾
    - t2i: 文件名包含 't2i' 且以 .html 结尾

    返回 OrderedDict:
    {
        model_name: {
            "i2t": i2t_html_path,
            "t2i": t2i_html_path,
        },
        ...
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

        # 必须同时有 i2t 和 t2i 才能做权重统计
        if i2t_path and t2i_path:
            models[name] = {"i2t": i2t_path, "t2i": t2i_path}

    return models


def fmt_pct(x):
    return f"{x:.2f}%" if x is not None else "–"


def generate_html(results, keywords, out_path: str):
    """
    results: {model_name: agg_dict or None}
    """
    if not results:
        print("没有任何模型结果可写入 HTML。")
        return

    model_names = list(results.keys())

    # 图表数据：画“使用 i2t 图片数为权重”的加权综合 acc
    labels = model_names
    weighted_top1 = [
        (results[m]["weighted_top1"] if results[m] is not None else None)
        for m in model_names
    ]
    weighted_top5 = [
        (results[m]["weighted_top5"] if results[m] is not None else None)
        for m in model_names
    ]

    html = []
    add = html.append

    add("<!DOCTYPE html>")
    add("<html lang='en'>")
    add("<head>")
    add("  <meta charset='UTF-8'>")
    add("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
    add("  <title>People-related Text Query Comparison (t2i, i2t-weighted)</title>")
    add("  <link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css' rel='stylesheet'>")
    add("  <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>")
    add("  <style>")
    add("    body { padding: 20px; font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }")
    add("    h1, h2, h3 { margin-top: 24px; }")
    add("    table { margin-top: 12px; }")
    add("    .model-table th, .model-table td { white-space: nowrap; }")
    add("  </style>")
    add("</head>")
    add("<body>")
    add("  <div class='container-fluid'>")
    add("    <h1>People-related Text Query Comparison (Text-to-Image, i2t-weighted)</h1>")
    add("    <p>统计所有 t2i 报告中，<strong>Query Text</strong> 包含下列关键词之一的查询：</p>")
    add("    <p><code>{}</code></p>".format(", ".join(keywords)))
    add("    <p>对于这些查询：</p>")
    add("    <ul>")
    add("      <li><strong>综合 acc（Macro）</strong>：对所有匹配查询的 Top-1 / Top-5 做<strong>等权平均</strong></li>")
    add("      <li><strong>加权综合 acc（Weighted）</strong>：使用对应类别在 <strong>i2t 报告中的 Image Count（图片数）</strong> 作为权重，对 Top-1 / Top-5 做加权平均</li>")
    add("    </ul>")

    # 图表
    add("    <h2>Weighted Overall Accuracy (i2t Image Count as Weight)</h2>")
    add("    <div class='row mb-4'>")
    add("      <div class='col-12'>")
    add("        <div class='card'>")
    add("          <div class='card-header'>加权综合 Top-1 / Top-5（权重 = i2t Image Count）</div>")
    add("          <div class='card-body'>")
    add("            <canvas id='weightedChart' height='90'></canvas>")
    add("          </div>")
    add("        </div>")
    add("      </div>")
    add("    </div>")

    # 表格
    add("    <h2>People-related Query Metrics (Table)</h2>")
    add("    <div class='table-responsive'>")
    add("    <table class='table table-striped table-bordered align-middle model-table'>")
    add("      <thead class='table-dark'>")
    add("        <tr>")
    add("          <th>Model</th>")
    add("          <th>#Matched Queries (t2i rows)</th>")
    add("          <th>Total Query Count (t2i)</th>")
    add("          <th>Macro Top-1</th>")
    add("          <th>Macro Top-5</th>")
    add("          <th>Weighted Top-1<br><small>(i2t Image Count)</small></th>")
    add("          <th>Weighted Top-5<br><small>(i2t Image Count)</small></th>")
    add("          <th>Sum of Weights<br><small>(Images used)</small></th>")
    add("        </tr>")
    add("      </thead>")
    add("      <tbody>")

    for m in model_names:
        agg = results[m]
        if agg is None:
            add("        <tr>")
            add(f"          <td>{m}</td>")
            add("          <td colspan='7' class='text-muted'>No people-related queries found or no valid i2t weights.</td>")
            add("        </tr>")
            continue

        add("        <tr>")
        add(f"          <td>{m}</td>")
        add(f"          <td>{agg['num_queries']}</td>")
        add(f"          <td>{agg['total_t2i_queries']}</td>")
        add(f"          <td>{fmt_pct(agg['macro_top1'])}</td>")
        add(f"          <td>{fmt_pct(agg['macro_top5'])}</td>")
        add(f"          <td>{fmt_pct(agg['weighted_top1'])}</td>")
        add(f"          <td>{fmt_pct(agg['weighted_top5'])}</td>")
        add(f"          <td>{agg['weighted_image_count']}</td>")
        add("        </tr>")

    add("      </tbody>")
    add("    </table>")
    add("    </div>")

    add("    <hr>")
    add("    <p class='text-muted'>Generated by people_query_comparison_i2t_weighted.py</p>")
    add("  </div>")

    # Chart.js
    add("  <script>")
    add("    const labels = " + json.dumps(labels) + ";")
    add("    const weightedTop1 = " + json.dumps(weighted_top1) + ";")
    add("    const weightedTop5 = " + json.dumps(weighted_top5) + ";")

    add("    const colors = [")
    add("      'rgba(54, 162, 235, 0.6)',")
    add("      'rgba(255, 99, 132, 0.6)'")
    add("    ];")
    add("    const borderColors = [")
    add("      'rgba(54, 162, 235, 1)',")
    add("      'rgba(255, 99, 132, 1)'")
    add("    ];")

    add("    const ctx = document.getElementById('weightedChart').getContext('2d');")
    add("    new Chart(ctx, {")
    add("      type: 'bar',")
    add("      data: {")
    add("        labels: labels,")
    add("        datasets: [")
    add("          {")
    add("            label: 'Weighted Top-1',")
    add("            data: weightedTop1,")
    add("            backgroundColor: colors[0],")
    add("            borderColor: borderColors[0],")
    add("            borderWidth: 1")
    add("          },")
    add("          {")
    add("            label: 'Weighted Top-5',")
    add("            data: weightedTop5,")
    add("            backgroundColor: colors[1],")
    add("            borderColor: borderColors[1],")
    add("            borderWidth: 1")
    add("          }")
    add("        ]")
    add("      },")
    add("      options: {")
    add("        responsive: true,")
    add("        scales: {")
    add("          y: {")
    add("            beginAtZero: true,")
    add("            max: 100,")
    add("            title: { display: true, text: 'Accuracy (%)' }")
    add("          }")
    add("        }")
    add("      }")
    add("    });")
    add("  </script>")

    add("</body>")
    add("</html>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"结果已写入: {out_path}")


def main():
    # 用法：
    #   python people_query_comparison_i2t_weighted.py [root_dir] [out_html] [keyword1 keyword2 ...]
    #
    # 示例：
    #   python people_query_comparison_i2t_weighted.py ./models people_i2t_weighted.html person people man woman
    #
    # 若不传关键词则使用 DEFAULT_KEYWORDS。
    if len(sys.argv) >= 2:
        root_dir = sys.argv[1]
    else:
        root_dir = "."

    if len(sys.argv) >= 3:
        out_html = sys.argv[2]
    else:
        out_html = "people_query_i2t_weighted_report.html"

    if len(sys.argv) >= 4:
        keywords = sys.argv[3:]
    else:
        keywords = DEFAULT_KEYWORDS

    models = find_model_reports(root_dir)
    if not models:
        print("在指定目录下没有找到既有 i2t 又有 t2i 报告的模型文件夹。")
        return

    results = OrderedDict()
    for model_name, paths in models.items():
        i2t_path = paths["i2t"]
        t2i_path = paths["t2i"]

        i2t_per_class = parse_i2t_per_class(i2t_path)
        people_rows = parse_t2i_people_rows(t2i_path, keywords=keywords)

        if not people_rows:
            results[model_name] = None
            continue

        agg = aggregate_people_with_i2t_weights(people_rows, i2t_per_class)
        results[model_name] = agg

    generate_html(results, keywords, out_html)


if __name__ == "__main__":
    main()
