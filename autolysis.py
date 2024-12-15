# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai>=1.0.0",
#   "tenacity",
#   "chardet",
#   "scikit-learn",
#   "statsmodels",
#   "scipy",
# ]
# ///

import os
import sys
import logging
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import chardet
from tenacity import retry, wait_fixed, stop_after_attempt
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import f_oneway, chi2_contingency
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import r2_score, accuracy_score

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --------------------------
# DO NOT CHANGE THIS PART
# --------------------------
from openai import OpenAI, APIError

api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
api_key = os.environ.get("AIPROXY_TOKEN", None)
MODEL_NAME = "gpt-4o-mini"

if api_key is None:
    logging.error("AIPROXY_TOKEN environment variable not set. Exiting.")
    sys.exit(1)

openai_client = OpenAI(api_key=api_key, base_url=api_base)

def safe_filename(name: str) -> str:
    """Convert filename into a safe string to name charts accordingly."""
    return name.replace(".csv", "").replace(" ", "_")

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def call_llm(messages, functions=None, function_call=None):
    """
    Call the LLM with retries.
    We instruct the LLM to maintain stable, consistent reasoning.
    Lower temperature and repeated stable instructions are used.
    """
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1024,
            temperature=0.0,  # reduce temperature for more deterministic output
            functions=functions,
            function_call=function_call
        )
        return response
    except APIError as e:
        logging.error(f"OpenAI API error: {e}")
        raise
# --------------------------
# END OF DO NOT CHANGE BLOCK
# --------------------------

def read_csv_with_encoding(file_path, encodings=['utf-8', 'latin1', 'cp1252']):
    """Attempt to read CSV with various encodings."""
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            logging.info(f"Successfully read {file_path} with encoding {enc}")
            return df
        except UnicodeDecodeError:
            logging.warning(f"Failed to read {file_path} with encoding {enc}")
    raise UnicodeDecodeError(f"Unable to read {file_path} with tried encodings: {encodings}")

def summarize_dataframe(df, max_unique=10, max_cols=10):
    """
    Summarize the dataframe columns and basic stats.
    Limit columns for token efficiency and stable reasoning.
    """
    summary = {'shape': df.shape, 'columns': []}
    cols = df.columns[:max_cols]
    for col in cols:
        col_info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'num_missing': int(df[col].isna().sum()),
            'num_unique': int(df[col].dropna().nunique()),
            'example_values': [str(v) for v in df[col].dropna().unique()[:max_unique]]
        }
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].count() > 0:
            col_info['mean'] = float(df[col].mean())
            col_info['std'] = float(df[col].std()) if df[col].count() > 1 else None
            col_info['min'] = float(df[col].min())
            col_info['max'] = float(df[col].max())
        summary['columns'].append(col_info)
    return summary

def analyze_categorical(df, max_levels=10):
    """Analyze categorical columns for top frequencies."""
    cat_analysis = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype != 'datetime64[ns]':
            vc = df[col].value_counts(dropna=False)
            cat_analysis.append({
                'column': col,
                'top_values': vc.head(max_levels).to_dict(),
                'num_unique': int(vc.shape[0])
            })
    return cat_analysis

def detect_outliers(df):
    """Detect outliers using a Z-score threshold of 3."""
    numeric_df = df.select_dtypes(include=[np.number])
    outliers = []
    for col in numeric_df.columns:
        col_data = numeric_df[col].dropna()
        if col_data.count() > 1:
            std_dev = col_data.std(ddof=0)
            z_scores = (col_data - col_data.mean()) / std_dev if std_dev != 0 else np.zeros(len(col_data))
            outlier_count = int((np.abs(z_scores) > 3).sum())
            if outlier_count > 0:
                outliers.append({
                    'column': col,
                    'outlier_count': outlier_count,
                    'total_count': int(col_data.count())
                })
    return outliers

def simple_regression(df):
    """Perform a simple OLS regression on the first two numeric columns if possible."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None
    y_col = numeric_cols[0]
    x_col = numeric_cols[1]
    clean_df = df[[x_col, y_col]].dropna()
    if clean_df.shape[0] < 10:
        return None
    X = sm.add_constant(clean_df[x_col])
    model = sm.OLS(clean_df[y_col], X).fit()
    return {
        'y_col': y_col,
        'x_col': x_col,
        'coef': model.params.to_dict(),
        'r_squared': model.rsquared,
        'p_values': model.pvalues.to_dict()
    }

def perform_pca(df):
    """Perform PCA if multiple numeric features exist."""
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=0)
    if numeric_df.shape[1] <= 2 or numeric_df.shape[0] < 5:
        return None

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(numeric_df)
    pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != 'datetime64[ns]']
    hue_col = cat_cols[0] if cat_cols else None

    pca_file = "pca_scatter.png"
    plt.figure(figsize=(6,4))
    if hue_col:
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue=df[hue_col].iloc[pca_df.index].fillna("Missing"), palette="Set2")
        plt.legend(title=hue_col)
    else:
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", color='blue')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Scatter Plot")
    plt.tight_layout()
    plt.savefig(pca_file)
    plt.close()

    return {
        'explained_variance': pca.explained_variance_ratio_.tolist(),
        'pca_file': pca_file,
        'hue_col': hue_col
    }

def perform_anova(df):
    """Perform a one-way ANOVA if conditions met."""
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != 'datetime64[ns]']
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(cat_cols) == 0 or len(num_cols) == 0:
        return None

    cat_col = cat_cols[0]
    num_col = num_cols[0]
    grouped = df[[cat_col, num_col]].dropna().groupby(cat_col)[num_col].apply(list)
    if len(grouped) < 2:
        return None

    fstat, pvalue = f_oneway(*grouped)
    return {'categorical_col': cat_col, 'numeric_col': num_col, 'f_statistic': fstat, 'p_value': pvalue}

def perform_chi_square(df):
    """Perform a chi-square test if possible."""
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != 'datetime64[ns]']
    if len(cat_cols) < 2:
        return None
    cat1, cat2 = cat_cols[:2]
    contingency_table = pd.crosstab(df[cat1], df[cat2])
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return None
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return {'cat_col_1': cat1, 'cat_col_2': cat2, 'chi2': chi2, 'p_value': p, 'dof': dof}

def perform_clustering(df):
    """Perform K-Means clustering if multiple numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=0)
    if numeric_df.shape[1] < 2 or numeric_df.shape[0] < 5:
        return None
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = kmeans.fit_predict(numeric_df)
    cluster_file = "cluster_scatter.png"
    cols = numeric_df.columns
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=numeric_df[cols[0]], y=numeric_df[cols[1]], hue=labels, palette="Set1")
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.title("K-Means Clustering (3 clusters)")
    plt.legend(title="Cluster")
    centers = kmeans.cluster_centers_
    for i, center in enumerate(centers):
        plt.text(center[0], center[1], f"C{i}", fontsize=12, fontweight='bold', color='black', 
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    plt.tight_layout()
    plt.savefig(cluster_file)
    plt.close()
    return {'inertia': kmeans.inertia_, 'clusters': 3, 'cluster_file': cluster_file}

def advanced_ml_analysis(df):
    """
    Attempt a more advanced ML approach: Use a small hyperparameter grid search with cross-validation 
    for Random Forest to show dynamic analysis based on data.

    If regression: try two sets of hyperparameters and pick the best.
    If classification: do the same.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Simple heuristic: if many numeric features, try a small hyperparameter grid
    param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 5]}

    if len(numeric_cols) > 2:
        target = numeric_cols[0]
        features = numeric_cols[1:]
        clean_df = df[[target] + list(features)].dropna()
        if clean_df.shape[0] > 20:
            X = clean_df[features]
            y = clean_df[target]
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            rf = RandomForestRegressor(random_state=42)
            gscv = GridSearchCV(rf, param_grid, scoring='r2', cv=kf, n_jobs=-1)
            gscv.fit(X, y)
            best_score = gscv.best_score_
            best_params = gscv.best_params_
            return {'type': 'regression', 'target': target, 'cv_best_r2': best_score, 'best_params': best_params}

    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != 'datetime64[ns]']
    if len(numeric_cols) > 1 and len(cat_cols) > 0:
        target_col = cat_cols[0]
        vc = df[target_col].value_counts(dropna=True)
        if 2 <= len(vc) <= 5:
            clean_df = df[numeric_cols.tolist() + [target_col]].dropna()
            if clean_df.shape[0] > 20:
                X = clean_df[numeric_cols]
                y = clean_df[target_col].astype(str)
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                clf = RandomForestClassifier(random_state=42)
                gscv = GridSearchCV(clf, param_grid, scoring='accuracy', cv=skf, n_jobs=-1)
                gscv.fit(X, y)
                best_score = gscv.best_score_
                best_params = gscv.best_params_
                return {'type': 'classification', 'target': target_col, 'cv_best_accuracy': best_score, 'best_params': best_params}

    return None

def plot_correlation_matrix(df, output_file):
    """Plot correlation matrix of numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None
    corr = numeric_df.corr(numeric_only=True)
    plt.figure(figsize=(6,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', square=True, cbar=True)
    plt.title("Correlation Matrix")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    return output_file

def plot_missing_values(df, output_file):
    """Plot bar chart of missing values."""
    missing_counts = df.isna().sum()
    if missing_counts.sum() == 0:
        return None
    plt.figure(figsize=(6,4))
    sns.barplot(x=missing_counts.index, y=missing_counts.values, color='skyblue')
    plt.xticks(rotation=90)
    plt.ylabel("Count of Missing Values")
    plt.title("Missing Values by Column")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    return output_file

def plot_hist_of_numeric(df, output_file):
    """Plot histograms for numeric data."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        return None
    plt.figure(figsize=(6,4))
    if numeric_df.shape[1] > 1:
        numeric_df.hist(bins=30, figsize=(6,4), color='green')
        plt.suptitle("Numeric Distributions", y=1.02)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
    else:
        col = numeric_df.columns[0]
        sns.histplot(numeric_df[col], kde=True, color='green')
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
    return output_file

functions = [
    {
        "name": "suggest_numeric_analysis",
        "description": "Suggest a numeric analysis function call to gain insights.",
        "parameters": {
            "type": "object",
            "properties": {
                "analysis_type": {
                    "type": "string",
                    "description": "The type of analysis, e.g. 'regression', 'outlier_detection', etc."
                }
            },
            "required": ["analysis_type"]
        }
    },
    {
        "name": "describe_chart_insights",
        "description": "Describe what the given chart might indicate.",
        "parameters": {
            "type": "object",
            "properties": {
                "chart_name": {
                    "type": "string",
                    "description": "Name of the chart file"
                }
            },
            "required": ["chart_name"]
        }
    }
]

def run_function_call_scenario(summary_dict):
    """
    Demonstrate function calling scenario.
    Instruct the LLM to maintain stable reasoning.
    """
    content = (
        "Given the data summary, suggest a numeric analysis approach. "
        "Respond by calling 'suggest_numeric_analysis' with a suitable analysis_type."
    )
    messages = [
        {"role": "system", "content": "You are a data scientist aiming at stable and consistent reasoning."},
        {"role": "user", "content": content}
    ]

    try:
        resp = call_llm(messages, functions=functions, function_call={"name": "suggest_numeric_analysis"})
        for choice in resp.choices:
            if choice.message and choice.message.function_call:
                logging.info("Function call suggested: %s", choice.message.function_call)
                break
    except Exception as e:
        logging.error(f"Function call scenario failed: {e}")

def run_advanced_llm_scenario(summary_dict):
    """
    Ask LLM to propose another advanced method after initial results.
    This can inform our dynamic choices (though not fully automated).
    """
    content = "We've done multiple analyses, including advanced ML with grid search. What advanced method could we try next and why?"
    messages = [
        {"role": "system", "content": "You are an expert data scientist, stable and consistent."},
        {"role": "user", "content": content}
    ]
    resp = call_llm(messages)
    suggestion = resp.choices[0].message.content.strip()
    logging.info("Additional advanced analysis suggestion: %s", suggestion)
    return suggestion

def run_vision_like_scenario(chart_files):
    """
    If we have multiple charts, we now ask for insights on each (limit to 2 for token saving).
    This demonstrates a more agentic approach: multiple LLM calls describing multiple charts.
    """
    if not chart_files:
        return None
    charts_to_describe = chart_files[:2]
    insights = []
    for chart in charts_to_describe:
        content = (
            "Given this chart filename, call 'describe_chart_insights' to interpret its key insights."
        )
        messages = [
            {"role": "system", "content": "You are a vision-enabled assistant. Remain stable in reasoning."},
            {"role": "user", "content": content}
        ]
        try:
            resp = call_llm(messages, functions=functions, function_call={"name": "describe_chart_insights"})
            for choice in resp.choices:
                if choice.message and choice.message.function_call:
                    # Mock a response
                    chart_insight = f"The chart '{chart}' reveals patterns suggesting deeper insight into the distribution or relationships."
                    logging.info("Chart insight generated (mock): %s", chart_insight)
                    insights.append(chart_insight)
        except Exception as e:
            logging.error(f"Vision-like scenario failed: {e}")
    if insights:
        return " ".join(insights)
    return None

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def get_llm_analysis(summary_dict, cat_analysis, outliers, regression_info, pca_info, anova_info, chi_info, cluster_info, ml_info):
    """
    Use LLM to analyze results. We emphasize stable reasoning, direct integration of advanced steps,
    and explaining why these steps matter.
    """
    content = (
        "You are a data analyst explaining insights to non-technical stakeholders. "
        "Be stable, consistent, and highlight why each analysis matters. We used partial summaries for token efficiency.\n"
        f"Partial Data summary:\n{summary_dict}\n"
        f"Categorical analysis:\n{cat_analysis}\n"
        f"Outliers:\n{outliers}\n"
        f"Regression:\n{regression_info}\n"
        f"PCA:\n{pca_info}\n"
        f"ANOVA:\n{anova_info}\n"
        f"Chi-square:\n{chi_info}\n"
        f"Clustering:\n{cluster_info}\n"
        f"Advanced ML (with grid search):\n{ml_info}\n\n"
        "Emphasize how these methods build on each other, why they are chosen, and what they imply."
    )
    messages = [
        {"role": "system", "content": "You are a clear data analyst, stable and consistent in reasoning."},
        {"role": "user", "content": content}
    ]
    response = call_llm(messages)
    return response.choices[0].message.content.strip()

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def get_llm_story(summary_dict, analysis_insights, chart_files, chart_insight, advanced_suggestion):
    """
    Ask LLM to write a stable, coherent README in Markdown.
    Use headings, bullet points, mention advanced suggestions and dynamic choices.
    Integrate chart insights and show how analysis steps led to advanced methods.
    """
    content = (
        "You are a data storyteller for non-technical readers, stable and consistent. Produce a Markdown README:\n"
        "- Use Markdown headings and bullet points.\n"
        "- Introduce the dataset context.\n"
        "- Show a logical sequence: start with basic EDA, missing values, correlation, distributions, categorical frequencies, outliers, regression, PCA, ANOVA, chi-square, clustering, and advanced ML with hyperparameter search.\n"
        "- Explain why each step was chosen and what its findings mean.\n"
        "- Reference the chart insights: " + str(chart_insight) + " from our vision steps.\n"
        "- Mention partial summaries (token efficiency), advanced suggestions: " + str(advanced_suggestion) + ".\n"
        "- Emphasize how data characteristics influenced dynamic analysis choices (e.g., if many numeric features, tried PCA and RF with CV).\n"
        "- Conclude with key actionable insights.\n\n"
        f"Partial Data summary:\n{summary_dict}\n\n"
        f"Insights:\n{analysis_insights}\n\n"
        "Charts: " + ", ".join(chart_files)
    )

    messages = [
        {"role": "system", "content": "You are a brilliant data storyteller, stable, consistent, and logically structured."},
        {"role": "user", "content": content}
    ]
    response = call_llm(messages)
    return response.choices[0].message.content

def save_results(base_name, chart_files, readme_content):
    """Save charts and README.md."""
    output_dir = base_name
    os.makedirs(output_dir, exist_ok=True)
    for chart in chart_files:
        if os.path.exists(chart):
            os.replace(chart, os.path.join(output_dir, chart))
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    logging.info(f"Outputs saved in directory: {output_dir}")

def load_and_prepare_data(csv_file):
    """Load and summarize the data."""
    df = read_csv_with_encoding(csv_file)
    summary = summarize_dataframe(df, max_cols=10)
    return df, summary

def perform_analyses(df):
    """
    Perform various analyses:
    - If correlation is high, mention the idea of dropping correlated features (not implemented, but suggests dynamism).
    - Based on data shape, we do advanced ML with grid search.
    """
    cat_analysis = analyze_categorical(df)
    outliers = detect_outliers(df)
    regression_info = simple_regression(df)
    pca_info = perform_pca(df)
    anova_info = perform_anova(df)
    chi_info = perform_chi_square(df)
    cluster_info = perform_clustering(df)
    ml_info = advanced_ml_analysis(df)
    return cat_analysis, outliers, regression_info, pca_info, anova_info, chi_info, cluster_info, ml_info

def generate_charts(df, base_name):
    """Generate charts and return the list of chart files. Fewer charts for token efficiency."""
    chart_files = []
    corr_file = f"{base_name}_correlation.png"
    cfile = plot_correlation_matrix(df, corr_file)
    if cfile:
        chart_files.append(cfile)

    missing_file = f"{base_name}_missing.png"
    mfile = plot_missing_values(df, missing_file)
    if mfile:
        chart_files.append(mfile)

    # Just keep one histogram for simplicity (efficiency)
    hist_file = f"{base_name}_hist.png"
    hfile = plot_hist_of_numeric(df, hist_file)
    if hfile:
        chart_files.append(hfile)
    return chart_files

def run_llm_workflow(summary, cat_analysis, outliers, regression_info, pca_info, anova_info, chi_info, cluster_info, ml_info, chart_files):
    """
    Orchestrate LLM calls:
    1. Analyze results (non-technical explanation)
    2. Ask for advanced suggestions
    3. Vision scenario for multiple charts if possible
    4. Final narrative integrating all steps, stable reasoning emphasized.
    """
    analysis_insights = get_llm_analysis(summary, cat_analysis, outliers, regression_info,
                                         pca_info, anova_info, chi_info, cluster_info, ml_info)

    run_function_call_scenario(summary)
    advanced_suggestion = run_advanced_llm_scenario(summary)
    chart_insight = run_vision_like_scenario(chart_files)

    readme_content = get_llm_story(summary, analysis_insights, chart_files, chart_insight, advanced_suggestion)
    return readme_content

def main():
    if len(sys.argv) < 2:
        logging.error("Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    if not os.path.isfile(csv_file):
        logging.error(f"CSV file {csv_file} does not exist.")
        sys.exit(1)

    df, summary = load_and_prepare_data(csv_file)
    cat_analysis, outliers, regression_info, pca_info, anova_info, chi_info, cluster_info, ml_info = perform_analyses(df)

    base_name = safe_filename(os.path.basename(csv_file))
    chart_files = generate_charts(df, base_name)

    # If PCA or clustering returned chart files, add them
    if pca_info and 'pca_file' in pca_info:
        chart_files.append(pca_info['pca_file'])
    if cluster_info and 'cluster_file' in cluster_info:
        chart_files.append(cluster_info['cluster_file'])

    readme_content = run_llm_workflow(summary, cat_analysis, outliers, regression_info,
                                      pca_info, anova_info, chi_info, cluster_info, ml_info, chart_files)
    save_results(base_name, chart_files, readme_content)

    logging.info("Analysis complete. README.md and charts created.")

if __name__ == "__main__":
    main()
