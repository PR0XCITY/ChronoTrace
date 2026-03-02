"""
fix_dashboard.py — patch script for ChronoTrace dashboard.py
Run once then delete.
"""
with open("dashboard.py", "r", encoding="utf-8") as f:
    txt = f.read()

# 1) Fix graph height
txt = txt.replace(
    "graph_h = 480 if not focus_ring else 420",
    "graph_h = 520 if not focus_ring else 460",
    1
)

# 2) Add dragmode=pan and fix margin (replace the exact margin + hoverlabel line)
OLD_MARGIN = '        margin=dict(l=10, r=10, t=10, b=30),\n        annotations=annotations,\n        hoverlabel=dict(bgcolor="#0d1520", font_size=11,'
NEW_MARGIN = '        margin=dict(l=10, r=10, t=15, b=35),\n        annotations=annotations,\n        dragmode="pan",\n        hoverlabel=dict(bgcolor="#0d1520", font_size=11,'
txt = txt.replace(OLD_MARGIN, NEW_MARGIN, 1)

# 3) Fix axis label color (tiny text "← Origin" was #374151 = nearly invisible)
txt = txt.replace(
    'font=dict(size=9, color="#374151"))),',
    'font=dict(size=9, color="#475569"))),',
    1
)

# 4) Relax blockchain eligibility — make it trigger on attack mode with DNA >= 20
# The current should_anchor() requires: DNA>=25 AND "Exit Imminent" AND "Quantum Vulnerable"
# We now lower the DNA bar to 20 and accept any suspicious stage (Pre-Cashout or higher)
OLD_ANCHOR = (
    '    return (\n'
    '        dna_score >= DNA_THRESHOLD\n'
    '        and laundering_stage == REQUIRED_STAGE\n'
    '        and pqc_status == REQUIRED_PQC_STATUS\n'
    '    )'
)
NEW_ANCHOR = (
    '    # Relaxed eligibility: DNA >= 20 and stage is Pre-Cashout or higher\n'
    '    # (or original strict condition also qualifies)\n'
    '    strict = (\n'
    '        dna_score >= DNA_THRESHOLD\n'
    '        and laundering_stage == REQUIRED_STAGE\n'
    '        and pqc_status == REQUIRED_PQC_STATUS\n'
    '    )\n'
    '    relaxed = (\n'
    '        dna_score >= 20.0\n'
    '        and laundering_stage in ("Pre-Cashout", "Exit Imminent", "Layering")\n'
    '    )\n'
    '    return strict or relaxed'
)
txt = txt.replace(OLD_ANCHOR, NEW_ANCHOR, 1)

with open("dashboard.py", "w", encoding="utf-8") as f:
    f.write(txt)

print("Patches applied successfully.")
