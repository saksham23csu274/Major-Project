"""
Self-updating HTML progress dashboard for presenting to your mentor/faculty.

Reuses the exact same PHASES definition and detection checks from
progress_report.py — nothing is duplicated. If you add a new check to
progress_report.py for a future milestone, this dashboard picks it up
automatically the next time you run it. Nothing here needs to change.

Usage:
    python scripts/progress_dashboard.py

Output:
    docs/progress_dashboard.html — open this in any browser. Re-run this
    script and refresh the page any time you want an updated view (e.g.
    right before your meeting, or live during it if you make a last-minute
    change).
"""

import datetime
from pathlib import Path

import progress_report as pr  # same directory — reuses PHASES + all checks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "docs" / "progress_dashboard.html"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ISL Translation System — Project Progress</title>
<style>
  :root {{
    --bg: #0f1419;
    --card-bg: #1a2029;
    --border: #2a3140;
    --text: #e6e9ef;
    --text-dim: #8b93a3;
    --accent: #4f8ef7;
    --done: #3ddc84;
    --pending: #3a4152;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    margin: 0;
    padding: 40px 24px;
  }}
  .container {{ max-width: 900px; margin: 0 auto; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
  .subtitle {{ color: var(--text-dim); font-size: 0.9rem; margin-bottom: 32px; }}

  .overall-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px;
    margin-bottom: 32px;
    text-align: center;
  }}
  .overall-pct {{
    font-size: 3.2rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
  }}
  .overall-label {{
    color: var(--text-dim);
    margin-top: 6px;
    font-size: 0.95rem;
  }}
  .overall-bar-track {{
    background: var(--pending);
    border-radius: 8px;
    height: 14px;
    margin-top: 20px;
    overflow: hidden;
  }}
  .overall-bar-fill {{
    background: linear-gradient(90deg, var(--accent), var(--done));
    height: 100%;
    border-radius: 8px;
    transition: width 0.3s ease;
  }}

  .phase-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }}
  .phase-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }}
  .phase-name {{ font-weight: 600; font-size: 1.05rem; }}
  .phase-pct {{ color: var(--text-dim); font-size: 0.9rem; }}
  .phase-bar-track {{
    background: var(--pending);
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
    margin-bottom: 14px;
  }}
  .phase-bar-fill {{
    background: var(--done);
    height: 100%;
    border-radius: 6px;
  }}
  .check-list {{ list-style: none; margin: 0; padding: 0; }}
  .check-item {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 5px 0;
    font-size: 0.92rem;
  }}
  .check-icon {{
    width: 18px; height: 18px;
    border-radius: 50%;
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem;
    font-weight: bold;
  }}
  .check-icon.done {{ background: var(--done); color: #0f1419; }}
  .check-icon.pending {{ background: var(--pending); color: var(--text-dim); }}
  .check-text.done {{ color: var(--text); }}
  .check-text.pending {{ color: var(--text-dim); }}

  .footer {{
    text-align: center;
    color: var(--text-dim);
    font-size: 0.8rem;
    margin-top: 32px;
  }}
</style>
</head>
<body>
<div class="container">
  <h1>Real-Time Sentence-Level ISL Translation System</h1>
  <div class="subtitle">Project progress — generated {timestamp}</div>

  <div class="overall-card">
    <div class="overall-pct">{overall_pct:.0f}%</div>
    <div class="overall-label">Overall project completion</div>
    <div class="overall-bar-track">
      <div class="overall-bar-fill" style="width: {overall_pct:.1f}%"></div>
    </div>
  </div>

  {phase_cards}

  <div class="footer">
    This page reflects the actual state of the project folder at generation time.
    Re-run <code>python scripts/progress_dashboard.py</code> and refresh to update.
  </div>
</div>
</body>
</html>
"""

PHASE_CARD_TEMPLATE = """
  <div class="phase-card">
    <div class="phase-header">
      <div class="phase-name">{name}</div>
      <div class="phase-pct">{pct:.0f}% ({done}/{total})</div>
    </div>
    <div class="phase-bar-track">
      <div class="phase-bar-fill" style="width: {pct:.1f}%"></div>
    </div>
    <ul class="check-list">
      {check_items}
    </ul>
  </div>
"""

CHECK_ITEM_TEMPLATE = """
      <li class="check-item">
        <span class="check-icon {state}">{icon}</span>
        <span class="check-text {state}">{desc}</span>
      </li>
"""


def build_dashboard_html() -> str:
    overall_earned = 0.0
    overall_total = 0.0
    phase_cards_html = []

    for phase in pr.PHASES:
        checks = phase["checks"]
        results = [(desc, bool(fn())) for desc, fn in checks]
        done_count = sum(1 for _, ok in results if ok)
        phase_pct = 100 * done_count / len(results) if results else 0

        weight = phase["weight"]
        overall_earned += weight * (phase_pct / 100)
        overall_total += weight

        check_items_html = "".join(
            CHECK_ITEM_TEMPLATE.format(
                state="done" if ok else "pending",
                icon="&#10003;" if ok else "",
                desc=desc,
            )
            for desc, ok in results
        )

        phase_cards_html.append(
            PHASE_CARD_TEMPLATE.format(
                name=phase["name"],
                pct=phase_pct,
                done=done_count,
                total=len(results),
                check_items=check_items_html,
            )
        )

    overall_pct = 100 * overall_earned / overall_total if overall_total else 0

    return HTML_TEMPLATE.format(
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        overall_pct=overall_pct,
        phase_cards="".join(phase_cards_html),
    )


def main() -> None:
    html = build_dashboard_html()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Dashboard written to: {OUTPUT_PATH}")
    print(f"Open it with: open {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
