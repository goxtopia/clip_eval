import html
import json
from typing import Dict, List, Any

def generate_html_report(output_path: str, metrics: Dict[str, float], per_class_stats: List[Dict[str, Any]], mode: str = "i2t"):
    def pct(x):
        return f"{x * 100:.2f}%"

    label_header = "Representative Label" if mode == "i2t" else "Query Text"
    count_header = "Image Count" if mode == "i2t" else "Query Count"
    
    rows_html = []
    for item in per_class_stats:
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(item['label'])}</td>"
            f"<td>{item['count']}</td>"
            f"<td>{pct(item['top1'])}</td>"
            f"<td>{pct(item['top5'])}</td>"
            "</tr>"
        )

    title = "CLIP Evaluation Report (Image-to-Text)" if mode == "i2t" else "CLIP Evaluation Report (Text-to-Image Search)"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ padding: 20px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        .metric-card {{ text-align: center; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; background-color: #f8f9fa; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #0d6efd; }}
        .metric-label {{ color: #6c757d; }}
        table {{ margin-top: 20px; }}
        th {{ cursor: pointer; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">{title}</h1>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card mb-4">
                    <div class="card-header">Global Performance</div>
                    <div class="card-body">
                        <canvas id="globalChart"></canvas>
                    </div>
                    <div class="card-footer text-muted">
                        Micro Top-1: {pct(metrics['global_top1'])} | Micro Top-5: {pct(metrics['global_top5'])}
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card mb-4">
                    <div class="card-header">Class-Averaged Performance</div>
                    <div class="card-body">
                        <canvas id="macroChart"></canvas>
                    </div>
                    <div class="card-footer text-muted">
                        Macro Top-1: {pct(metrics['macro_top1'])} | Macro Top-5: {pct(metrics['macro_top5'])}
                    </div>
                </div>
            </div>
        </div>

        <h3>Per-Class Breakdown</h3>
        <p>Sorted by Label. Click headers to sort.</p>
        <input class="form-control mb-3" id="searchInput" type="text" placeholder="Search...">
        
        <table class="table table-striped table-hover" id="resultsTable">
            <thead class="table-dark">
                <tr>
                    <th onclick="sortTable(0)">{label_header}</th>
                    <th onclick="sortTable(1, true)">{count_header}</th>
                    <th onclick="sortTable(2, true)">Top-1 Acc</th>
                    <th onclick="sortTable(3, true)">Top-5 Acc</th>
                </tr>
            </thead>
            <tbody id="tableBody">
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>

    <script>
        // Charts
        const ctxGlobal = document.getElementById('globalChart').getContext('2d');
        new Chart(ctxGlobal, {{
            type: 'bar',
            data: {{
                labels: ['Top-1', 'Top-5'],
                datasets: [{{
                    label: 'Global (Micro) Accuracy',
                    data: [{metrics['global_top1'] * 100}, {metrics['global_top5'] * 100}],
                    backgroundColor: ['rgba(54, 162, 235, 0.5)', 'rgba(75, 192, 192, 0.5)'],
                    borderColor: ['rgba(54, 162, 235, 1)', 'rgba(75, 192, 192, 1)'],
                    borderWidth: 1
                }}]
            }},
            options: {{ scales: {{ y: {{ beginAtZero: true, max: 100 }} }} }}
        }});

        const ctxMacro = document.getElementById('macroChart').getContext('2d');
        new Chart(ctxMacro, {{
            type: 'bar',
            data: {{
                labels: ['Macro Top-1', 'Macro Top-5', 'Weighted Top-1', 'Weighted Top-5'],
                datasets: [{{
                    label: 'Class-Averaged Accuracy',
                    data: [
                        {metrics['macro_top1'] * 100}, {metrics['macro_top5'] * 100},
                        {metrics['weighted_top1'] * 100}, {metrics['weighted_top5'] * 100}
                    ],
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.5)', 'rgba(255, 99, 132, 0.5)',
                        'rgba(255, 159, 64, 0.5)', 'rgba(255, 159, 64, 0.5)'
                    ],
                    borderColor: [
                        'rgba(255, 99, 132, 1)', 'rgba(255, 99, 132, 1)',
                        'rgba(255, 159, 64, 1)', 'rgba(255, 159, 64, 1)'
                    ],
                    borderWidth: 1
                }}]
            }},
            options: {{ scales: {{ y: {{ beginAtZero: true, max: 100 }} }} }}
        }});

        // Search
        document.getElementById("searchInput").addEventListener("keyup", function() {{
            var value = this.value.toLowerCase();
            var rows = document.querySelectorAll("#tableBody tr");
            rows.forEach(row => {{
                var text = row.cells[0].textContent.toLowerCase();
                row.style.display = text.indexOf(value) > -1 ? "" : "none";
            }});
        }});

        // Sort
        var currentSortCol = -1;
        var currentSortAsc = true;

        function sortTable(n, isNumeric=false) {{
            const table = document.getElementById("resultsTable");
            const tbody = document.getElementById("tableBody");
            const rows = Array.from(tbody.getElementsByTagName("tr"));

            // Toggle direction
            if (currentSortCol === n) {{
                currentSortAsc = !currentSortAsc;
            }} else {{
                currentSortCol = n;
                currentSortAsc = true;
            }}

            // Sort array
            rows.sort((a, b) => {{
                let x = a.getElementsByTagName("td")[n].textContent;
                let y = b.getElementsByTagName("td")[n].textContent;

                if (isNumeric) {{
                    x = parseFloat(x.replace("%", ""));
                    y = parseFloat(y.replace("%", ""));
                }} else {{
                    x = x.toLowerCase();
                    y = y.toLowerCase();
                }}

                if (x < y) return currentSortAsc ? -1 : 1;
                if (x > y) return currentSortAsc ? 1 : -1;
                return 0;
            }});

            // Re-append
            const fragment = document.createDocumentFragment();
            rows.forEach(row => fragment.appendChild(row));
            tbody.appendChild(fragment);
        }}
    </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[Info] HTML report written to {output_path}")
