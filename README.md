#  Mutual Fund Style Drift Detector

> Detecting when Indian mutual fund managers quietly shift their investment style away from their declared SEBI category — using unsupervised machine learning on AMFI public portfolio disclosures.

---

##  Problem Statement

When you invest in a **"Large Cap"** mutual fund, SEBI mandates that the fund manager keeps at least 80% of the portfolio in large-cap stocks. But fund managers sometimes gradually shift holdings toward mid or small cap stocks to chase higher returns — without changing the fund's name or marketing. This is called **style drift**, and retail investors are exposed to more risk than they signed up for.

This project builds an **automated style drift detection system** that:
- Ingests monthly portfolio disclosures for 12 mutual funds across 2 AMCs
- Classifies every holding by its official SEBI market cap category
- Engineers a behavioural fingerprint for each fund each month
- Clusters funds using K-Means to discover natural groupings
- Flags funds whose behaviour is drifting away from their declared category over time

---

##  Dataset

| Source | Description | Coverage |
|--------|-------------|----------|
| [AMFI Monthly Portfolio Disclosures](https://www.amfiindia.com/online-center/portfolio-disclosure) | Complete stock holdings per fund per month | Jan 2023 – May 2026 |
| [AMFI Cap Classification Lists](https://www.amfiindia.com/otherdata/categorisation-of-stocks) | Official SEBI large/mid/small cap stock rankings | Jul 2022 – Dec 2025 (7 windows) |

**Funds analysed:** 6 SBI funds + 6 Kotak funds across all 6 SEBI equity categories

| Fund | Declared Category | Months |
|------|-------------------|--------|
| SBI Large Cap Fund | Large Cap | 40 |
| SBI Large and Midcap Fund | Large & Mid Cap | 40 |
| SBI Midcap Fund | Mid Cap | 40 |
| SBI MultiCap Fund | Multi Cap | 40 |
| SBI Flexicap Fund | Flexi Cap | 40 |
| SBI Smallcap Fund | Small Cap | 40 |
| Kotak Large Cap Fund | Large Cap | 19 |
| Kotak Large & Midcap Fund | Large & Mid Cap | 19 |
| Kotak Midcap Fund | Mid Cap | 19 |
| Kotak Multicap Fund | Multi Cap | 19 |
| Kotak Flexicap Fund | Flexi Cap | 19 |
| Kotak Small Cap Fund | Small Cap | 19 |

**Final dataset:** 22,461 holding-level rows · 12 funds · up to 40 months

---

##  Methodology

### Phase 1 — Data Pipeline (`style_drift_detector.ipynb`)
- Scraped monthly Excel portfolio disclosures from AMFI for SBI and Kotak AMCs
- Handled inconsistent file formats, legacy S3 URLs, and JavaScript-gated downloads
- Built a unified `holdings_master.csv` with 22,461 rows and 10 columns
- Extracted true `statement_date` from file contents rather than trusting link labels

### Phase 2 — Cap Classification Join (`style_drift_detector.ipynb`)
- Downloaded all 7 AMFI cap classification lists (Jul 2022 → Dec 2025)
- Applied a **window-aware join** — each holding matched to the classification list valid *as of its statement date*, not today's list
- Handled 2026 holdings (beyond Dec 2025 list) by proxying the most recent available classification — documented transparently as a known limitation
- Result: 97% of rows successfully classified (only 3% genuinely unclassifiable bonds/ETFs)

| Classification Result | Rows |
|----------------------|------|
| Exact window match | 17,900 |
| Proxied (2026 rows) | 3,888 |
| Unclassified (bonds/ETFs/foreign) | 673 |

### Phase 3 — Feature Engineering & ML (`style_drift_ml.ipynb`)

**Feature matrix** — one row per fund per month (354 rows × 5 features):

| Feature | Description |
|---------|-------------|
| `pct_large_cap` | % of fund AUM in large-cap stocks that month |
| `pct_mid_cap` | % of fund AUM in mid-cap stocks that month |
| `pct_small_cap` | % of fund AUM in small-cap stocks that month |
| `top10_concentration` | % of AUM held in the fund's top 10 stocks |
| `sector_hhi` | Herfindahl-Hirschman Index of sector concentration |

**K-Means Clustering (K=5)**
- K selected via elbow method — clear bend at K=5
- Features standardised with StandardScaler before clustering
- K-Means discovered groupings that closely mirror SEBI's own legal categories — completely unsupervised

**Cluster alignment with declared categories:**

| Cluster | Dominant Category | Purity |
|---------|------------------|--------|
| Cluster 0 | Mid Cap | 100% |
| Cluster 1 | Large Cap + Flexi Cap | ~97% |
| Cluster 2 | Large & Mid Cap + Multi Cap | ~100% |
| Cluster 3 | Small Cap | 100% |
| Cluster 4 | Outliers / Anomalies | 6 rows |

**PCA** — compressed 5 features to 2 dimensions capturing **79.6% of total variance**

**Drift Score** — Euclidean distance of each fund-month from its cluster centroid in PCA space. Higher = further from typical behaviour for its declared category.

---

##  Results

### PCA Scatter Plot — Declared Categories vs Discovered Clusters

<img width="1350" height="607" alt="image" src="https://github.com/user-attachments/assets/8e371008-1904-4cc9-a694-404513745d7c" />


> *Left: how funds declared themselves. Right: what K-Means discovered from actual holdings data — with no labels provided.*

---

### Style Drift Over Time — SBI Funds

<img width="1211" height="902" alt="image" src="https://github.com/user-attachments/assets/d4d322e7-c036-4370-92c9-516c172d5d6a" />


### Style Drift Over Time — Kotak Funds

<!-- 
  INSERT IMAGE HERE
  File: outputs/drift_over_time.png  (bottom half — Kotak panel)
  Notebook: style_drift_ml.ipynb
  Generated by: Cell P3-5 (drift over time chart cell)
  Description: Line chart showing drift score per month for all 6 Kotak funds, Nov 2024 – May 2026
-->

---

###  Key Findings

**1. Kotak Flexicap Fund — Active Drift Signal (Most Significant)**
Kotak Flexicap shows a clear, sustained upward drift trend from mid-2025 through May 2026, reaching the 1.0 threshold at the last data point. This is 12 consecutive months of increasing distance from its cluster centroid — textbook style drift, not noise.

**2. SBI Large Cap Fund & SBI Smallcap Fund — Historical Anomalies**
Both funds show extreme drift spikes in early 2023 (max scores: 4.595 and 4.678 respectively), coinciding with SEBI's post-recategorisation enforcement period when fund managers were actively rebalancing portfolios. Scores normalised by mid-2023.

**3. Kotak Large & Midcap Fund — Gradual Upward Trend**
Steady climb from 0.2 (Nov 2024) to 0.85 (May 2026). Not yet above threshold but on a clear upward trajectory.

**4. K-Means validated SEBI's category boundaries**
Mid Cap and Small Cap funds formed 100%-pure clusters with zero contamination — confirming these categories have truly distinct portfolio behaviours that the algorithm can detect without any labels.

### Drift Score Summary

| Fund | Mean Drift | Max Drift | Std Dev | Status |
|------|-----------|----------|---------|--------|
| SBI Large Cap Fund | 0.543 | 4.595 | 0.698 |  Historic spike (2023) |
| SBI Smallcap Fund | 0.457 | 4.678 | 0.694 |  Historic spike (2023) |
| SBI Midcap Fund | 0.402 | 3.926 | 0.599 |  Historic spike (2023) |
| SBI Flexicap Fund | 0.672 | 2.501 | 0.415 |  Recent breach (2026) |
| Kotak Flexicap Fund | 0.529 | 0.999 | 0.228 |  Active drift trend |
| Kotak Large & Midcap Fund | 0.707 | 0.846 | 0.068 |  Rising trend |
| Kotak Large Cap Fund | 0.265 | 0.363 | 0.055 |  Stable |
| Kotak Midcap Fund | 0.194 | 0.355 | 0.114 |  Stable |

---

##  Repository Structure

```
StyleDrift_Project/
│
├── style_drift_detector.ipynb    ← Phase 1 & 2: data pipeline + cap classification
├── style_drift_ml.ipynb          ← Phase 3: feature engineering + ML
│
├── raw_data/
│   ├── amfi_portfolio/           ← Monthly Excel disclosures (not uploaded — large files)
│   ├── amfi_cap_list/            ← 7 AMFI cap classification Excel files
│   └── nse_benchmarks/           ← NSE index constituents
│
├── processed/
│   ├── holdings_master.csv       ← 22,461 rows, raw parsed holdings
│   ├── holdings_with_cap.csv     ← + cap_category and cap_match_type columns
│   └── feature_matrix_final.csv  ← 354 rows, 5 features + drift scores
│
└── outputs/
    ├── elbow_plot.png            ← K selection via elbow method
    ├── pca_scatter.png           ← Side-by-side PCA visualisation
    └── drift_over_time.png       ← Drift score line charts per fund
```

---

##  Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| Pandas | Data wrangling and feature engineering |
| Scikit-learn | K-Means clustering, PCA, StandardScaler |
| Matplotlib | All visualisations |
| Google Colab | Development environment |
| Google Drive | Persistent storage across sessions |
| tqdm | Progress bars for long-running joins |

---

##  Known Limitations

1. **Only 2 AMCs** — SBI and Kotak. HDFC, ICICI, and Nippon India had JavaScript-gated portals that blocked direct download; expanding coverage is a planned Week 2 task.
2. **2026 cap classification proxied** — AMFI's Jun 2026 list publishes in July 2026. All Jan–May 2026 holdings use the Dec 2025 list as proxy. Market cap rankings shift slowly; this is a minor limitation.
3. **Drift threshold of 1.0 is empirical** — set based on the distribution of scores across all funds and months. A statistically derived threshold (e.g. mean + 2σ per category) is a planned improvement.
4. **No Sharpe ratio or rolling return signals yet** — the current drift score uses portfolio composition only. Adding return-based metrics as corroborating evidence is Phase 5.

---

##  What's Next (Week 2)

- [ ] Add HDFC and Nippon India AMC data (expanding to ~25 funds)
- [ ] Add Sharpe ratio and rolling return volatility as corroborating drift signals
- [ ] Statistically derived drift threshold per category (mean + 2σ)
- [ ] Streamlit dashboard for interactive fund-level drift exploration
- [ ] Backtesting — do high drift scores predict future underperformance?

---

##  Author

**Divyadharshini** — B.Tech Computer Science · PES University, Electronic City, Bangalore  
GitHub: [@divyadharshini-1306](https://github.com/divyadharshini-1306)  
Email: divyadharshini.mb.13@gmail.com

---

*Data sourced entirely from public AMFI disclosures. This project is for educational and portfolio purposes only and does not constitute financial advice.*
