"""
fix_blockchain_ui.py — patch blockchain UI text
"""
with open("dashboard.py", "r", encoding="utf-8") as f:
    txt = f.read()

# Fix the "not eligible" info message to reflect relaxed criteria
old_info = (
    "            st.info(\r\n"
    "                \"\u26d3\ufe0f Blockchain anchoring activates when **Exit Imminent** stage \"\r\n"
    "                \"is detected with DNA score \u2265 25 and Quantum Vulnerable status.\",\r\n"
    "                icon=\"\U0001f510\",\r\n"
    "            )"
)
new_info = (
    "            st.info(\r\n"
    "                \"\u26d3\ufe0f Blockchain anchoring activates when **DNA \u2265 20** and stage is \"\r\n"
    "                \"**Layering, Pre-Cashout, or Exit Imminent**. Run an attack simulation to trigger.\",\r\n"
    "                icon=\"\U0001f510\",\r\n"
    "            )"
)
txt = txt.replace(old_info, new_info, 1)

# Also update the success message to include simulation mode info
old_success = '        st.success("\u2714\ufe0f Alert hash anchored to Ethereum Sepolia.", icon="\u26d3\ufe0f")'
new_success = (
    '        ar_mode = (st.session_state.anchor_result or {}).get("mode", "Simulation")\r\n'
    '        mode_icon = "\U0001f7e2" if ar_mode == "Live Sepolia" else "\U0001f7e1"\r\n'
    '        st.success(f"\u2714\ufe0f Alert anchored! Mode: **{ar_mode}** {mode_icon}", icon="\u26d3\ufe0f")'
)
txt = txt.replace(old_success, new_success, 1)

with open("dashboard.py", "w", encoding="utf-8") as f:
    f.write(txt)

print("Blockchain UI patches applied.")
