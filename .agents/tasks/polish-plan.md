# Visual Polish Implementation Plan

This plan targets presentation-only improvements to the Credit Risk Analytics Platform.
No functionality, prediction logic, model files, thresholds, or CSV export logic will change.

Files to modify:
- `app.py` (lines 1-1568)

---

## 1. Tighten typography and standardize spacing

**What to change**
- Increase heading `line-height` to `1.2` for crisper appearance
- Increase `letter-spacing` on main header `h1` to `0.3px`
- Replace scattered `st.markdown('<br>', unsafe_allow_html=True)` calls with consistent spacing helpers (CSS margin classes or `st.markdown('<div class="spacing-md"></div>', unsafe_allow_html=True)`)

**Where in the file**
- CSS block (lines 13-203): Update `.main-header h1`, `.section-header`, add `.spacing-sm`, `.spacing-md`, `.spacing-lg` classes
- Page content sections (lines ~412, ~535, ~615, ~883, ~995, ~1100, ~1240): Replace `<br>` with `<div class="spacing-md"></div>` or similar

**What NOT to touch**
- Do not remove `<br>` from inside metric or card HTML strings where they are intentionally used for layout
- Do not modify prediction or scoring functions
- Do not change any `st.subheader`, `st.metric`, or `st.dataframe` calls

---

## 2. Replace emoji navigation icons and section markers

**What to change**
- Replace emoji in sidebar radio navigation (lines ~372-380):
  - `"📊 Overview"` → `"● Overview"`
  - `"👤 Customer"` → `"› Customer Assessment"`
  - `"📈 Portfolio"` → `"› Portfolio Scoring"`
  - `"🔍 Insights"` → `"› Model Insights"`
- Replace numbered emoji in section headers (e.g., `1️⃣ Credit Profile` → `1. Credit Profile`)
- Remove `icon=` parameter from `st.info`, `st.warning`, `st.success`, `st.error` where the emoji adds visual noise (e.g., keep only where it aids scanning: `⚠️` for warnings that need immediate attention)

**Where in the file**
- Sidebar navigation (lines ~372-380)
- Customer Assessment sections (lines ~563, ~579, ~596, ~613, ~630)
- Overview page info boxes (lines ~395, ~410, ~518)
- Customer Assessment info/error boxes (lines ~552, ~708, ~718, ~753, ~765, ~773)
- Portfolio Scoring info/error boxes (lines ~859, ~883, ~1034)
- Model Insights info/error boxes (lines ~1213, ~1223, ~1322, ~1343)

**What NOT to touch**
- Do not change any section content or labels (e.g., "Credit Profile", "Repayment History")
- Do not modify the `st.radio` options list structure or keys
- Do not change the form field labels or help text

---

## 3. Improve chart labels and axis styling

**What to change**
- Add `labelsize=10` to all `ax.tick_params` calls
- Add `color="#e6f1ff"` and `fontweight="600"` to `ax.set_title` calls
- Add units in parentheses to axis labels where missing (e.g., `ax.set_xlabel("Risk Category (Category)", color="#94a3b8")`)
- Remove top and right spines from bar/histogram charts: `ax.spines['top'].set_visible(False)`, `ax.spines['right'].set_visible(False)`
- Ensure bar value annotations use consistent format and position

**Where in the file**
- Risk Distribution chart (lines ~452-485)
- Probability Distribution chart (lines ~488-518)
- Probability Visualization chart (lines ~681-712)
- Portfolio Scoring Risk Distribution chart (lines ~979-1009)
- Model Insights Feature Importance chart (lines ~1270-1320)

**What NOT to touch**
- Do not change any data transformations or chart data
- Do not alter chart dimensions (`figsize=`)
- Do not modify axis data ranges or thresholds

---

## 4. Apply `.stCard` class and verify metric styling

**What to change**
- Wrap major sections in `<div class='stCard'>...</div>` to use the existing CSS class
- Verify `[data-testid='stMetric']` padding (should be `1rem`) and `border-radius` (should be `10px`) match across sidebar and main content

**Where in the file**
- Overview metrics section (lines ~421-445)
- Customer Assessment results section (lines ~643-665)
- Portfolio Scoring summary metrics (lines ~965-985)
- Model Insights performance metrics (lines ~1202-1215)

**What NOT to touch**
- Do not change the metric key names or values
- Do not modify any `st.metric` arguments or labels
- Do not alter the CSS class name or base styles

---

## 5. Hover states and transitions

**What to change**
- Ensure button hover `transform: translateY(-1px)` and `box-shadow` are present for primary buttons
- Add `transition: border-color 0.2s ease` to `[data-testid='stMetric']`
- Add `cursor: pointer` to `.stDownloadButton button`

**Where in the file**
- CSS block (lines 13-203): Add to `[data-testid='stMetric']` and `.stDownloadButton button` selectors

**What NOT to touch**
- Do not change button text or labels
- Do not modify any form field or interactive element behavior

---

## 6. Responsive layout fixes

**What to change**
- Replace any fixed pixel widths with `max-width` or percentages
- Verify 3-column and 4-column layouts use `st.columns([3, 1])` or similar proportional values
- Add `max-width: 100%` to chart containers in CSS

**Where in the file**
- CSS block (lines 13-203): Add responsive overrides in a `@media` query block
- Chart rendering sections: Ensure `use_container_width=True` for dataframes and consistent container sizing

**What NOT to touch**
- Do not change column layout logic (`st.columns(3)` → `st.columns(4)`)
- Do not modify any chart data or rendering logic

---

## 7. Minor clutter removal

**What to change**
- Remove emoji from main header `<h1>` (lines ~360-370): use plain text title per page without emoji
- Remove redundant `<br>` after `st.divider()` calls (e.g., lines ~404, ~529, ~560, ~642, ~679, ~873, ~950, ~1191, ~1267)
- Clean up `st.caption(icon=...)` calls (line ~727, ~1365): remove unsupported `icon=` parameter

**Where in the file**
- Main header (lines ~360-370)
- Spacing after dividers (multiple locations)
- Caption calls (lines ~727, ~1365)

**What NOT to touch**
- Do not change any text content or labels
- Do not modify any `st.divider()` or section structure
- Do not remove any `st.caption` calls, only remove the `icon=` parameter

---

## Verification

After applying changes, run:
```bash
cd c:\Users\vishnu vardhan\Desktop\Credit-RISK && python -m streamlit run app.py
```

Expected outcomes:
- App launches without errors or warnings
- All four pages render with consistent spacing and professional appearance
- Charts have cleaner axes, consistent labels, and removed top/right spines
- No emoji in navigation or section headers
- Hover states work on buttons and metrics
- Layout degrades gracefully on smaller screens

No functional changes to prediction, scoring, file I/O, or CSV export are expected.