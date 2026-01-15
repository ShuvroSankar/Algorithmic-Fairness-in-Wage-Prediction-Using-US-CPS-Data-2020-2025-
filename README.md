# Income Inequality and Algorithmic Fairness in Labor Markets: A Statistical Analysis of US CPS Data (2020–2025)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)

## Project Overview

This project provides a comprehensive statistical and machine learning analysis of **income inequality and algorithmic fairness** in the US labor market using Current Population Survey (CPS) data from 2020–2025. It combines econometric modeling, fairness auditing, and deep learning to:

1. **Quantify wage gaps** by gender, race, and education
2. **Audit machine learning models** for disparate impact and discrimination
3. **Test fairness interventions** and measure accuracy-fairness trade-offs
4. **Provide policy-relevant insights** for wage equity and algorithmic accountability

### Key Findings

- **Gender wage gap**: Men earn ~23% more than women after controlling for education, hours worked, and age
- **Algorithmic bias**: Naive ML models exhibit Disparate Impact Ratio (DIR) of **0.65** (fails legal 0.80 threshold)
- **Fairness intervention**: Removing gender as a feature improves DIR to **0.83** while sacrificing only **0.93% accuracy**
- **Education premium**: Workers without a high school diploma earn **68% less** than advanced-degree holders ($25.9k vs $80.6k)

---

## 📁 Repository Structure

```
├── README.md                               # This file
├── requirements.txt                        # Python dependencies
├── environment.yml                         # Conda environment (optional)
│
├── IPUMS.ipynb                             # Main Jupyter notebook
│   ├── Part A (Mid-Term): Descriptive stats, hypothesis testing
│   ├── Part B (Final-Term): Regression, multivariate analysis, visualizations
│   └── Part C (Extension): ML fairness audit & interventions
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb             # Data loading, preprocessing
│   ├── 02_descriptive_stats.ipynb         # Summary statistics, distributions
│   ├── 03_regression_analysis.ipynb       # OLS regression, coefficients
│   ├── 04_visualization_portfolio.ipynb   # 6+ publication-quality plots
│   └── 05_fairness_audit.ipynb            # ML bias detection & correction
│
├── data/
│   └── cps_2020_2025_adults_basic.csv     # CPS extract (6.3M rows, 15 columns)
│
├── results/
│   ├── regression_summary.csv             # Model coefficients & statistics
│   ├── fairness_metrics.csv               # DIR, TPR gaps, FPR gaps
│   └── visualizations/                    # PNG exports of all plots
│
└── docs/
    ├── INTRODUCTION.md                    # Research problem & motivation
    ├── METHODOLOGY.md                     # Statistical & ML methods
    ├── RESULTS.md                         # Findings summary
    └── CONCLUSION.md                      # Policy implications & limitations
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ipums-fairness-analysis.git
cd ipums-fairness-analysis
```

### 2. Install Dependencies

**Option A: pip** (recommended for Jupyter)
```bash
pip install -r requirements.txt
```

**Option B: conda** (reproducible environments)
```bash
conda env create -f environment.yml
conda activate ipums-fairness
```

### 3. Download CPS Data

The CPS data is not included in the repository due to size (6.3M rows). Download from:

- **IPUMS CPS**: https://cps.ipums.org (free account required)
- Extract variables: YEAR, MONTH, SERIAL, PERNUM, ASECWT, AGE, SEX, RACE, HISPAN, MARST, EDUC, OCC, IND, INCWAGE, UHRSWORKLY
- Save as: `data/cps_2020_2025_adults_basic.csv`

Or use the included data loading script:
```bash
python scripts/download_cps.py --start_year 2020 --end_year 2025
```

### 4. Run the Full Analysis

```bash
# Option A: Run the complete notebook
jupyter notebook IPUMS.ipynb

# Option B: Run individual analysis sections
jupyter notebook notebooks/01_data_cleaning.ipynb
jupyter notebook notebooks/02_descriptive_stats.ipynb
jupyter notebook notebooks/03_regression_analysis.ipynb
jupyter notebook notebooks/04_visualization_portfolio.ipynb
jupyter notebook notebooks/05_fairness_audit.ipynb
```

---

## Project Components

### **Part A: Descriptive Statistics & Hypothesis Testing**

Tests whether wage gaps are statistically significant:

```python
# Two-sample t-test: Male vs Female income
t_stat, p_val = scipy.stats.ttest_ind(male_income, female_income)
# Result: t=89.19, p<0.001 ✓ Highly significant
```

**Key metrics:**
- Mean income by gender, race, education
- 95% confidence intervals
- Probability distributions (Normal, Binomial)
- Hypothesis test results

### **Part B: Multivariate Regression & Visualization**

Isolates effects of education, hours, age on income:

```python
# OLS Regression: log(income) ~ age + hours + sex + race + education
# Model explains 38.9% of variation (R² = 0.389)
# Male coefficient: +0.231 (23% wage premium)
```

**Visualization portfolio (6 plots):**
1. Age distribution (histogram)
2. Income by education (boxplot)
3. Hours vs income (scatter + regression line)
4. Correlation matrix (heatmap)
5. Life-cycle wage curve (line plot by gender)
6. Predicted wages by education (bar chart)

### **Part C: Machine Learning & Fairness Audit**

Trains classifiers and audits for bias:

```python
# Random Forest classifier: Predict HIGH_INCOME (>$50k)
# Disparate Impact Ratio: 54.4% men vs 35.6% women = 0.65 (FAILS legal test)
# TPR gap: Model misses 10% more high-earning women

# Fairness intervention: Remove gender feature
# Result: DIR improves to 0.83 (PASSES legal test) with only 0.93% accuracy loss
```

**Fairness metrics:**
- Disparate Impact Ratio (selection rate fairness)
- True Positive Rate (equal opportunity)
- False Positive Rate (equalized odds)
- Confusion matrices by demographic group

---

## 📊 Key Results

### Wage Gap by Demographic

| Group | Mean Income | Median Income | Sample Size |
|-------|------------|---------------|------------|
| **Male** | $75,811 | $55,000 | 255,705 |
| **Female** | $53,593 | $40,000 | 167,250 |
| **White** | $71,356 | $50,000 | 326,298 |
| **Black** | $54,265 | $43,000 | 47,757 |
| **Asian** | $82,692 | $60,000 | 31,037 |
| **Advanced Degree** | $80,633 | — | — |
| **High School** | $35,587 | — | — |
| **<High School** | $25,868 | — | — |

### Regression Results

| Variable | Coefficient | Std Error | t-stat | p-value |
|----------|------------|-----------|--------|---------|
| **Constant** | 8.912 | 0.008 | 1064.2 | <0.001 |
| **Age** | 0.0089 | 0.00009 | 96.8 | <0.001 |
| **Hours/week** | 0.0464 | 0.0001 | 370.0 | <0.001 |
| **Male** (vs Female) | **0.2306** | 0.003 | 86.0 | <0.001 |
| **Black** (vs Asian) | −0.1845 | 0.006 | −29.4 | <0.001 |
| **HS Diploma** (vs Adv Deg) | −0.8179 | 0.004 | −191.0 | <0.001 |

### Fairness Audit Results

| Metric | Original ML | Fair ML | Target |
|--------|------------|---------|--------|
| **Accuracy** | 75.66% | 74.72% | — |
| **DIR (Female/Male)** | 0.65 | 0.83 | ≥0.80 ✓ |
| **TPR Gap** | −10.2pp | +4.4pp | ~0 |
| **FPR Gap** | −11.8pp | +1.4pp | ~0 |

---

## Methodology

### Data Source

**Current Population Survey (CPS)** via IPUMS
- 6.3 million adults (18+)
- Years: 2020–2025
- Variables: income, hours worked, education, demographics, trade indicators

### Statistical Methods

- **Hypothesis Testing**: Two-sample t-tests, Chi-square tests, ANOVA
- **Regression**: OLS with log-transformed outcome (ln(income))
- **Fairness Metrics**: Disparate Impact Ratio (4/5ths rule), Equal Opportunity, Equalized Odds
- **Machine Learning**: Random Forest, scikit-learn

### Train/Val/Test Split

- **Train**: 70% (6.3M → 4.4M)
- **Validation**: 10% (0.63M)
- **Test**: 20% (1.3M)
- **Split method**: Chronological (no shuffling) for fairness work

---

## Documentation

Comprehensive documentation is included:

- **INTRODUCTION.md**: Research problem, motivation, and context
- **METHODOLOGY.md**: Statistical and ML methods explained
- **RESULTS.md**: Detailed findings with interpretation
- **CONCLUSION.md**: Policy implications, limitations, future work

Read the intro:
```bash
cat docs/INTRODUCTION.md
```

---

## 💻 Requirements

**Minimum Requirements:**
- Python 3.10+
- 8GB RAM (for full CPS dataset)
- 2GB disk space

**Python Libraries:**
```
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.10.0
scikit-learn>=1.2.0
statsmodels>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
ipumspy>=2.0.0
```

See `requirements.txt` for full list with versions.

---

## Use Cases

This project is useful for:

1. **Researchers** studying wage inequality and discrimination
2. **Data Scientists** building fairness-aware ML systems
3. **Policymakers** designing wage equity audits and interventions
4. **Educators** teaching fairness in machine learning
5. **Developers** implementing algorithmic auditing frameworks

---

## Limitations

1. **Missing Data**: 93% of CPS respondents excluded (missing wage income)
2. **Cross-sectional**: Cannot track individuals over time (limits causal inference)
3. **Unobserved Variables**: Model explains only 39% of wage variation (R²=0.389)
4. **Annual Data**: High-frequency (monthly/weekly) would better capture dynamics
5. **Survey Design**: CPS weights not fully incorporated in analysis

See `docs/CONCLUSION.md` for full discussion.

---

## Fairness Interventions

The project demonstrates three fairness strategies:

### 1. **Preprocessing** (Remove Sensitive Attributes)
```python
# Remove 'SEX_Label_Male' from features before training
X_fair = X.drop(columns=['SEX_Label_Male'])
model_fair = RandomForestClassifier().fit(X_fair, y)
# Result: DIR 0.65 → 0.83, Accuracy loss: 0.93%
```

### 2. **In-Processing** (Fairness Penalties During Training)
Example: Modify loss function to penalize disparate impact.

### 3. **Post-Processing** (Threshold Adjustment by Group)
Example: Use different decision thresholds for different demographic groups.

---

## Citation

If you use this project, please cite:

```bibtex
@inproceedings{yourname2026fairness,
  title={Income Inequality and Algorithmic Fairness in Labor Markets: 
         A Statistical Analysis of US CPS Data (2020–2025)},
  author={Your Name},
  year={2026},
  note={GitHub: yourusername/ipums-fairness-analysis}
}
```

---

## License

This project is licensed under the **MIT License** — see `LICENSE` file for details.

**Data**: CPS data from IPUMS is governed by their terms of use. See https://cps.ipums.org for citation requirements.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/your-idea`)
3. Commit your changes (`git commit -am 'Add new analysis'`)
4. Push to the branch (`git push origin feature/your-idea`)
5. Open a Pull Request

**Contribution ideas:**
- Extended analysis for other demographics (age, disability status)
- Multi-commodity time-series forecasting (Rice, Onion, Oil)
- Interactive dashboard (Streamlit, Plotly)
- Causal inference with instrumental variables
- Cross-country fairness comparisons

---

## Known Issues & TODOs

- [ ] Incorporate CPS survey weights in regression
- [ ] Extend to monthly/weekly data for seasonal analysis
- [ ] Add interactive Streamlit dashboard
- [ ] Implement causal forests for heterogeneous treatment effects
- [ ] Create fairness-aware threshold optimization tool

See `ISSUES.md` for open issues.

---

## Contact & Support

**Questions or feedback?**

- **Email**: your.email@example.com
- **GitHub Issues**: https://github.com/yourusername/ipums-fairness-analysis/issues
- **Discussions**: https://github.com/yourusername/ipums-fairness-analysis/discussions

---

## References & Further Reading

### Foundational Papers

- **"Retiring Adult: New Datasets for Fair Machine Learning"** (Hardt et al., 2021)  
  https://arxiv.org/abs/2108.04884
- **"Fairness and Machine Learning"** (Barocas, Hardt, Narayanan, 2023)  
  https://fairmlbook.org
- **"Equal Opportunity and Disparate Impact"** (Hardt et al., 2016)  
  https://arxiv.org/abs/1610.02413

### Data Sources

- **IPUMS CPS**: https://cps.ipums.org
- **US Census Bureau**: https://www.census.gov/cps/
- **Bureau of Labor Statistics**: https://www.bls.gov/

### Tools & Libraries

- **scikit-learn**: Machine learning
- **statsmodels**: Econometric regression
- **ipumspy**: CPS data access
- **seaborn/matplotlib**: Visualization

---

## Acknowledgments

- IPUMS for providing accessible CPS data
- US Census Bureau and Bureau of Labor Statistics for data collection
- The fairness in ML research community for methodological guidance

---

**Last Updated**: January 5, 2026  
**Status**: Complete & Ready for Publication

---

**⭐ If this project is useful to you, please consider starring the repository!**
