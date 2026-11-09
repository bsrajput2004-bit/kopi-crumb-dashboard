# Kopi & Crumb – consumer survey analytics dashboard

A Streamlit dashboard that tests whether **Kopi & Crumb**, a Nanyang kopi bar with butter croissants and kaya toast in JLT, Dubai, is a sustainable business.

It uses a 22-question consumer survey (5,000 synthetic respondents with intentional noise and outliers), cleans the data, and runs:

1. **Classification** (logistic regression, decision tree, random forest, gradient boosting): who will come (Q14)
2. **Clustering** (K-means): customer segments and high / medium / low conversion tiers
3. **Association rule mining** (Apriori): which 4–7 products to launch first (Q15)
4. **Regression** (linear regression, random forest): expected monthly spend (Q18)

The **Verdict** and **Sustainability** tabs combine the results with break-even maths. Every business assumption can be changed in the sidebar.

## Files
| File | What it is |
| --- | --- |
| `app.py` | The dashboard |
| `requirements.txt` | Python packages Streamlit installs |
| `kopi_crumb_survey_5000.csv` | The synthetic survey data (5,000 rows) |
| `generate_survey.py` | How the synthetic data was made (optional) |

## Using real survey data
Export the real survey with the same column names as `kopi_crumb_survey_5000.csv` and upload it in the sidebar. Every tab reruns on the real data.

## Run on your own computer (optional)
```
pip install -r requirements.txt
streamlit run app.py
```
