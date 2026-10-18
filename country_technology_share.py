#!/usr/bin/env python3
"""Create a 100% stacked country-by-technology share figure from the three-layer rule-based classified CSV.

Each publication-country pair is counted once for N_p, and each publication-
country-technology pair is counted once in the stacks. Country lists accept
semicolon or comma separators; technology labels are semicolon-separated so
commas inside labels remain intact.
"""
import argparse
import csv
import os
import re
from collections import defaultdict

from reportlab.lib.colors import HexColor, white, black
from reportlab.pdfgen import canvas

TECH_ORDER = [
    ("D1", "AI / ML", "#1f77b4"),
    ("D2", "Remote sensing", "#ff7f0e"),
    ("D3", "IoT & sensors", "#009e73"),
    ("D4", "Cloud / FMIS", "#cc79a7"),
    ("D5", "Robotics", "#e69f00"),
    ("D6", "Decision support", "#56b4e9"),
    ("D7", "Blockchain", "#999999"),
]


def split_countries(value):
    """Split country lists and remove repeats case-insensitively, preserving order."""
    parts = re.split(r"[;,]", value or "")
    found, seen = [], set()
    for part in parts:
        country = part.strip()
        key = country.casefold()
        if country and key not in seen:
            seen.add(key)
            found.append(country)
    return found


def split_technologies(value):
    """Split semicolon-separated technology labels and remove within-row repeats."""
    parts = (value or "").split(";")
    found, seen = [], set()
    for part in parts:
        label = part.strip()
        if not label:
            continue
        code = label.split(None, 1)[0].upper()
        key = code if re.fullmatch(r"D[1-7]", code) else label.casefold()
        if key not in seen:
            seen.add(key)
            found.append(key if re.fullmatch(r"D[1-7]", key) else label)
    return found


def aggregate(input_csv):
    country_papers = defaultdict(set)
    domain_papers = defaultdict(lambda: defaultdict(set))
    with open(input_csv, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"countries", "tech_domains"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=1):
            doi = (row.get("DOI") or "").strip().casefold()
            paper_id = f"doi:{doi}" if doi else f"row:{row_number}"
            countries = split_countries(row.get("countries", ""))
            techs = split_technologies(row.get("tech_domains", ""))
            for country in countries:
                country_papers[country].add(paper_id)
                for tech in techs:
                    code = tech[:2] if tech[:2] in {f"D{i}" for i in range(1, 8)} else ""
                    if code:
                        domain_papers[country][code].add(paper_id)

    # Rank countries by distinct paper count, matching the reference's N_p ordering.
    top = sorted(country_papers, key=lambda c: (-len(country_papers[c]), c.casefold()))[:20]
    result = []
    for country in top:
        counts = {code: len(domain_papers[country].get(code, set())) for code, _, _ in TECH_ORDER}
        total_assignments = sum(counts.values())
        shares = {code: (100.0 * count / total_assignments if total_assignments else 0.0)
                  for code, count in counts.items()}
        result.append({"country": country, "n_papers": len(country_papers[country]),
                       "domain_counts": counts, "domain_shares": shares,
                       "n_domain_assignments": total_assignments})
    return result


def write_summary(path, data):
    fields = ["rank", "country", "n_papers", "n_domain_assignments"]
    for code, _, _ in TECH_ORDER:
        fields.extend([f"{code}_papers", f"{code}_share_pct"])
    with open(path, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, item in enumerate(data, start=1):
            row = {"rank": rank, "country": item["country"], "n_papers": item["n_papers"],
                   "n_domain_assignments": item["n_domain_assignments"]}
            for code, _, _ in TECH_ORDER:
                row[f"{code}_papers"] = item["domain_counts"][code]
                row[f"{code}_share_pct"] = f"{item['domain_shares'][code]:.4f}"
            writer.writerow(row)


def draw_pdf(path, data):
    width, height = 900, 700
    c = canvas.Canvas(path, pagesize=(width, height), pageCompression=1)
    c.setTitle("Technology domain share by country")
    left, plot_width = 170, 630
    right_x = left + plot_width
    top_y, row_step, bar_h = 601, 23.5, 17
    axis_y = 119

    # Header and plot labels.
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 17)
    c.drawCentredString(width / 2, 665, "Technology-domain composition across countries")
    c.setFont("Helvetica-Oblique", 12)
    c.drawString(right_x + 14, 625, "N")
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(right_x + 22, 622, "p")

    # Country bars, ordered by publication count.
    for index, item in enumerate(data):
        y = top_y - index * row_step
        c.setFillColor(black)
        c.setFont("Helvetica", 11)
        c.drawRightString(left - 12, y + 4, item["country"])
        x = left
        for code, _label, color in TECH_ORDER:
            share = item["domain_shares"][code]
            seg_w = plot_width * share / 100.0
            if seg_w > 0:
                c.setFillColor(HexColor(color))
                c.setStrokeColor(white)
                c.setLineWidth(0.55)
                c.rect(x, y, seg_w, bar_h, fill=1, stroke=1)
                x += seg_w
        c.setFillColor(black)
        c.setFont("Helvetica", 11)
        c.drawString(right_x + 14, y + 4, f"{item['n_papers']:,}")

    # Axis and grid/ticks.
    c.setStrokeColor(HexColor("#333333"))
    c.setLineWidth(0.8)
    c.line(left, axis_y, right_x, axis_y)
    for tick in range(0, 101, 10):
        x = left + plot_width * tick / 100.0
        c.setStrokeColor(HexColor("#dddddd"))
        c.setLineWidth(0.45)
        # Keep the stacked bars clean, as in the reference; ticks stay on the axis.
        c.setStrokeColor(HexColor("#333333"))
        c.setLineWidth(0.8)
        c.line(x, axis_y, x, axis_y - 4)
        c.setFillColor(black)
        c.setFont("Helvetica", 9)
        c.drawCentredString(x, axis_y - 16, str(tick))
    c.saveState()
    c.translate(36, (axis_y + top_y) / 2)
    c.rotate(90)
    c.setFont("Helvetica", 12)
    c.drawCentredString(0, 0, "Country")
    c.restoreState()
    c.setFont("Helvetica", 12)
    c.drawCentredString(left + plot_width / 2, 83,
                        "Share of technology assignments within country (%)")

    # Two-row legend closely following the reference figure.
    legend_positions = [(145, 54), (345, 54), (545, 54), (745, 54),
                        (245, 31), (445, 31), (645, 31)]
    for (code, label, color), (x, y) in zip(TECH_ORDER, legend_positions):
        c.setFillColor(HexColor(color))
        c.setStrokeColor(white)
        c.rect(x, y, 18, 10, fill=1, stroke=1)
        c.setFillColor(black)
        c.setFont("Helvetica", 10)
        c.drawString(x + 24, y + 1, label)
    c.setFont("Helvetica", 8)
    c.setFillColor(HexColor("#444444"))
    c.drawCentredString(width / 2, 10,
        "N_p is distinct publications; each paper-country-technology assignment is counted once.")
    c.showPage()
    c.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, help="Three-layer rule-based classified CSV with countries and tech_domains columns.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of countries to plot (default: 20).")
    parser.add_argument("--output-dir", default=None, help="Output folder (defaults to input folder)")
    parser.add_argument("--stem", default="fig_country_domain_share", help="Output file name stem")
    args = parser.parse_args()
    if args.top_n < 1:
        parser.error("--top-n must be at least 1")
    if not os.path.isfile(args.input_csv):
        parser.error(f"Input CSV does not exist: {args.input_csv}")
    out_dir = os.path.abspath(args.output_dir or os.path.dirname(os.path.abspath(args.input_csv)))
    os.makedirs(out_dir, exist_ok=True)
    data = aggregate(args.input_csv)[:args.top_n]
    pdf_path = os.path.join(out_dir, args.stem + ".pdf")
    csv_path = os.path.join(out_dir, args.stem + "_data.csv")
    draw_pdf(pdf_path, data)
    write_summary(csv_path, data)
    print(f"PDF: {pdf_path}")
    print(f"Summary: {csv_path}")
    print(f"Countries plotted: {len(data)}")
    print("Top countries:", ", ".join(f"{x['country']} ({x['n_papers']:,})" for x in data))


if __name__ == "__main__":
    main()
