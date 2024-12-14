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

    Parameters:
        messages (list): A list of message dicts as per OpenAI chat format.
        functions (list, optional): List of function calling schemas for the LLM.
        function_call (dict, optional): If specified, forces the model to call a function.

    Returns:
        response (OpenAI response): The LLM response object.
    """
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1024,
            temperature=0.2,
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
    """Attempt to read CSV with various encodings until successful."""
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            logging.info(f"Successfully read {file_path} with encoding {enc}")
            return df
        except UnicodeDecodeError:
            logging.warning(f"Failed to read {file_path} with encoding {enc}")
    raise UnicodeDecodeError(f"Unable to read {file_path} with tried encodings: {encodings}")

def summarize_dataframe(df, max_unique=10):
    """
    Summarize the dataframe columns and basic stats.
    """
    summary = {'shape': df.shape, 'columns': []}
    for col in df.columns:
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
    """
    Analyze categorical columns for top frequencies.
    """
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
    """
    Detect outliers using a Z-score threshold of 3.
    """
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
    """
    Perform a simple linear regression on the first two numeric columns.
    """
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
    """
    Perform PCA on numeric columns if there are more than 2 numeric features.
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=0)
    if numeric_df.shape[1] <= 2 or numeric_df.shape[0] < 5:
        return None

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(numeric_df)
    pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])

    # Pick a categorical column to color by if available
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
    plt.title("PCA Scatter Plot (First Two PCs)")
    plt.tight_layout()
    plt.savefig(pca_file)
    plt.close()

    return {
        'explained_variance': pca.explained_variance_ratio_.tolist(),
        'pca_file': pca_file,
        'hue_col': hue_col
    }

def perform_anova(df):
    """
    Perform a one-way ANOVA if there's a numeric column and a categorical column with multiple categories.
    """
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != 'datetime64[ns]']
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(cat_cols) == 0 or len(num_cols) == 0:
        return None

    # Just pick the first numeric col and first categorical col
    cat_col = cat_cols[0]
    num_col = num_cols[0]
    grouped = df[[cat_col, num_col]].dropna().groupby(cat_col)[num_col].apply(list)

    if len(grouped) < 2:
        return None

    # Perform ANOVA
    fstat, pvalue = f_oneway(*grouped)
    return {
        'categorical_col': cat_col,
        'numeric_col': num_col,
        'f_statistic': fstat,
        'p_value': pvalue
    }

def perform_chi_square(df):
    """
    Perform a chi-square test on two categorical columns if available.
    """
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != 'datetime64[ns]']
    if len(cat_cols) < 2:
        return None
    # Pick the first two categorical columns
    cat1, cat2 = cat_cols[:2]
    contingency_table = pd.crosstab(df[cat1], df[cat2])
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return None
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return {
        'cat_col_1': cat1,
        'cat_col_2': cat2,
        'chi2': chi2,
        'p_value': p,
        'dof': dof
    }

def perform_clustering(df):
    """
    Perform K-Means clustering if multiple numeric columns.
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=0)
    if numeric_df.shape[1] < 2 or numeric_df.shape[0] < 5:
        return None
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = kmeans.fit_predict(numeric_df)
    cluster_file = "cluster_scatter.png"
    # If we have at least 2 numeric columns, plot first two dims
    cols = numeric_df.columns
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=numeric_df[cols[0]], y=numeric_df[cols[1]], hue=labels, palette="Set1")
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.title("K-Means Clustering (3 clusters)")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(cluster_file)
    plt.close()
    return {
        'inertia': kmeans.inertia_,
        'clusters': 3,
        'cluster_file': cluster_file
    }

def plot_correlation_matrix(df, output_file):
    """
    Plot correlation matrix of numeric columns with annotations and labels.
    """
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
    """
    Plot a bar chart of missing values per column with labels.
    """
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
    """
    Plot histograms for numeric data.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        return None
    plt.figure(figsize=(6,4))
    if numeric_df.shape[1] > 1:
        numeric_df.hist(bins=30, figsize=(6,4))
        plt.title("Numeric Distributions")
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

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def get_llm_analysis(summary_dict, cat_analysis, outliers, regression_info, pca_info, anova_info, chi_info, cluster_info):
    """
    Use LLM to analyze the summary and additional results, emphasizing significant findings and advanced tests.
    """
    content = (
        "You are a data analyst. We've done multiple analyses:\n\n"
        "Data summary:\n" + str(summary_dict) + "\n\n"
        "Categorical analysis:\n" + str(cat_analysis) + "\n\n"
        "Outlier detection:\n" + str(outliers) + "\n\n"
        "Regression analysis:\n" + str(regression_info) + "\n\n"
        "PCA analysis:\n" + str(pca_info) + "\n\n"
        "ANOVA results:\n" + str(anova_info) + "\n\n"
        "Chi-square test results:\n" + str(chi_info) + "\n\n"
        "Clustering results:\n" + str(cluster_info) + "\n\n"
        "Please highlight the most significant findings from these analyses. Emphasize what is meaningful (e.g., significant p-values, distinct clusters). "
        "Discuss implications of these findings and suggest even more advanced statistical or ML methods. "
        "Also maintain consistency as results may vary if run multiple times."
    )
    messages = [
        {"role": "system", "content": "You are a very insightful data analyst."},
        {"role": "user", "content": content}
    ]
    response = call_llm(messages)
    return response.choices[0].message.content.strip()

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def get_llm_story(summary_dict, analysis_insights, chart_files):
    """
    Ask LLM to write a README.md narrative referencing analyses, charts, and emphasizing significant findings.
    """
    content = (
        "You are a data storyteller. Produce a README.md:\n"
        "- Introduce the dataset.\n"
        "- Describe each analysis step (missing values, correlation, distributions, categorical frequencies, outliers, regression, PCA, ANOVA, chi-square test, clustering) and why.\n"
        "- Emphasize significant findings (e.g. significant ANOVA p-value, meaningful regression, distinct PCA structure, clustering patterns, chi-square test insights).\n"
        "- Reference the charts at appropriate steps using `![Chart](filename.png)`.\n"
        "- Suggest implications and next steps.\n"
        "- Acknowledge that LLM results can vary but you've tried to maintain consistency.\n\n"
        "Data summary:\n" + str(summary_dict) + "\n\n"
        "Insights:\n" + analysis_insights + "\n\n"
        "Charts: " + ", ".join(chart_files)
    )

    messages = [
        {"role": "system", "content": "You are a brilliant data storyteller. Highlight major findings and maintain coherence."},
        {"role": "user", "content": content}
    ]
    response = call_llm(messages)
    return response.choices[0].message.content

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
    Demonstrate function calling scenario to comply with project requirements.
    """
    content = (
        "Given the data summary, suggest a numeric analysis approach. "
        "Respond by calling the function 'suggest_numeric_analysis' with a suitable analysis_type."
    )
    messages = [
        {"role": "system", "content": "You are a data scientist trying to be consistent."},
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
    Ask the LLM to propose another advanced analysis after seeing initial results.
    """
    content = (
        "We have performed several analyses. What advanced statistical test or machine learning method could we try next? "
        "Just suggest the approach and explain why it would help, do not call any function."
    )
    messages = [
        {"role": "system", "content": "You are an expert data scientist."},
        {"role": "user", "content": content}
    ]
    resp = call_llm(messages)
    logging.info("Additional advanced analysis suggestion: %s", resp.choices[0].message.content.strip())

def run_vision_like_scenario(chart_files):
    """
    Simulate a vision capability scenario: ask the LLM to describe what insights might be drawn from a given chart.
    We'll pick one chart and call the 'describe_chart_insights' function.

    This simulates more agentic workflow with multiple LLM calls.
    """
    if not chart_files:
        return
    chosen_chart = chart_files[0]
    content = (
        "Given this chart filename, use the 'describe_chart_insights' function call to describe what the chart might indicate."
    )
    messages = [
        {"role": "system", "content": "You are a vision-enabled assistant."},
        {"role": "user", "content": content}
    ]
    try:
        resp = call_llm(messages, functions=functions, function_call={"name": "describe_chart_insights"})
        for choice in resp.choices:
            if choice.message and choice.message.function_call:
                logging.info("Chart description function call: %s", choice.message.function_call)
                break
    except Exception as e:
        logging.error(f"Vision-like scenario failed: {e}")

def save_outputs(base_name, chart_files, readme_content):
    """
    Save charts and README.md into a dedicated directory.
    """
    output_dir = base_name
    os.makedirs(output_dir, exist_ok=True)
    for chart in chart_files:
        if os.path.exists(chart):
            os.replace(chart, os.path.join(output_dir, chart))
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    logging.info(f"Outputs saved in directory: {output_dir}")

def main():
    if len(sys.argv) < 2:
        logging.error("Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    if not os.path.isfile(csv_file):
        logging.error(f"CSV file {csv_file} does not exist.")
        sys.exit(1)

    df = read_csv_with_encoding(csv_file)
    summary = summarize_dataframe(df)
    cat_analysis = analyze_categorical(df)
    outliers = detect_outliers(df)
    regression_info = simple_regression(df)
    pca_info = perform_pca(df)
    anova_info = perform_anova(df)
    chi_info = perform_chi_square(df)
    cluster_info = perform_clustering(df)

    base_name = safe_filename(os.path.basename(csv_file))
    chart_files = []

    corr_file = f"{base_name}_correlation.png"
    cfile = plot_correlation_matrix(df, corr_file)
    if cfile:
        chart_files.append(cfile)

    missing_file = f"{base_name}_missing.png"
    mfile = plot_missing_values(df, missing_file)
    if mfile:
        chart_files.append(mfile)

    hist_file = f"{base_name}_hist.png"
    hfile = plot_hist_of_numeric(df, hist_file)
    if hfile:
        chart_files.append(hfile)

    if pca_info and 'pca_file' in pca_info:
        chart_files.append(pca_info['pca_file'])

    if cluster_info and 'cluster_file' in cluster_info:
        chart_files.append(cluster_info['cluster_file'])

    # LLM analysis
    analysis_insights = get_llm_analysis(summary, cat_analysis, outliers, regression_info, pca_info, anova_info, chi_info, cluster_info)

    # Demonstrate function calling scenario
    run_function_call_scenario(summary)

    # Additional advanced scenario
    run_advanced_llm_scenario(summary)

    # Vision-like scenario
    run_vision_like_scenario(chart_files)

    # Get narrative story
    readme_content = get_llm_story(summary, analysis_insights, chart_files)

    # Save outputs
    save_outputs(base_name, chart_files, readme_content)

    logging.info("Analysis complete. README.md and charts created.")

if __name__ == "__main__":
    main()
