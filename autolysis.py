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
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.api as sm

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


# ===========================
# Helper functions
# ===========================
def safe_filename(name: str) -> str:
    # Convert filename into a safe string to name charts accordingly
    return name.replace(".csv", "").replace(" ", "_")

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def call_llm(messages, functions=None, function_call=None):
    """Call the LLM with retries. If function_call is specified, it attempts function calling."""
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

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def read_csv_with_encoding(file_path, encodings=['utf-8', 'latin1', 'cp1252']):
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            logging.info(f"Successfully read {file_path} with encoding {enc}")
            return df
        except UnicodeDecodeError:
            logging.warning(f"Failed to read {file_path} with encoding {enc}")
    raise UnicodeDecodeError(f"Unable to read {file_path} with tried encodings: {encodings}")

def summarize_dataframe(df, max_unique=10):
    """Summarize each column, including numeric stats and examples."""
    summary = {}
    summary['shape'] = df.shape
    summary['columns'] = []
    for col in df.columns:
        col_info = {}
        col_info['name'] = col
        col_info['dtype'] = str(df[col].dtype)
        col_info['num_missing'] = int(df[col].isna().sum())
        unique_vals = df[col].dropna().unique()
        col_info['num_unique'] = int(len(unique_vals))
        col_info['example_values'] = [str(v) for v in unique_vals[:max_unique]]

        # Basic stats if numeric
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].count() > 0:
            col_info['mean'] = float(df[col].mean())
            col_info['std'] = float(df[col].std()) if df[col].count() > 1 else None
            col_info['min'] = float(df[col].min())
            col_info['max'] = float(df[col].max())
        summary['columns'].append(col_info)
    return summary

def analyze_categorical(df, max_levels=10):
    """Get top frequency counts for categorical columns."""
    cat_analysis = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype != 'datetime64[ns]':
            vc = df[col].value_counts(dropna=False)
            top_vals = vc.head(max_levels).to_dict()
            cat_analysis.append({
                'column': col,
                'top_values': top_vals,
                'num_unique': int(vc.shape[0])
            })
    return cat_analysis

def detect_outliers(df):
    """Simple outlier detection using Z-scores for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    outlier_info = []
    for col in numeric_df.columns:
        col_data = numeric_df[col].dropna()
        if col_data.count() > 1:
            z_scores = (col_data - col_data.mean()) / col_data.std(ddof=0) if col_data.std(ddof=0) != 0 else np.zeros(len(col_data))
            outlier_count = int((np.abs(z_scores) > 3).sum())
            if outlier_count > 0:
                outlier_info.append({
                    'column': col,
                    'outlier_count': outlier_count,
                    'total_count': int(col_data.count())
                })
    return outlier_info

def simple_regression(df):
    """Perform a simple linear regression on the first two numeric columns."""
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

def plot_correlation_matrix(df, output_file):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None
    corr = numeric_df.corr(numeric_only=True)
    plt.figure(figsize=(6,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', square=True, cbar=True)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    return output_file

def plot_missing_values(df, output_file):
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
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        return None
    plt.figure(figsize=(6,4))
    if numeric_df.shape[1] > 1:
        numeric_df.hist(bins=30, figsize=(6,4))
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
    else:
        col = numeric_df.columns[0]
        sns.histplot(numeric_df[col], kde=True, color='green')
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
    return output_file

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def get_llm_analysis(summary_dict, cat_analysis, outliers, regression_info):
    """Ask the LLM for deeper insights including categorical, outliers, and regression."""
    content = (
        "You are a data analyst. I have a dataset summary and additional analysis:\n\n"
        f"Data summary: {summary_dict}\n\n"
        f"Categorical analysis: {cat_analysis}\n\n"
        f"Outlier detection: {outliers}\n\n"
        f"Regression analysis: {regression_info}\n\n"
        "Please provide thorough insights on numeric and categorical aspects, highlight outliers found, interpret the regression, "
        "and suggest deeper analyses for the future."
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
    Request a coherent and structured Markdown narrative.
    """
    content = (
        "You are a data storyteller. I performed the following analyses on the dataset:\n"
        "- Missing value analysis\n"
        "- Correlation analysis\n"
        "- Distribution histograms\n"
        "- Categorical frequency counts\n"
        "- Outlier detection (Z-score)\n"
        "- Simple linear regression between two numeric variables\n\n"
        f"Dataset summary:\n{summary_dict}\n\n"
        f"Insights:\n{analysis_insights}\n\n"
        "Chart files: " + ", ".join(chart_files) + "\n\n"
        "Produce a README.md in Markdown with:\n"
        "- Introduction of the dataset\n"
        "- Structured overview of each analysis step, with clear transitions\n"
        "- Insights discovered (categorical distributions, outliers, regression interpretation)\n"
        "- Implications or next steps\n"
        "- Include charts using `![Chart](filename.png)`"
    )

    messages = [
        {"role": "system", "content": "You are a brilliant data storyteller with attention to detail."},
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
    }
]

def run_function_call_scenario(summary_dict):
    """Demonstrate function calling scenario."""
    content = (
        "Given the data summary, suggest a numeric analysis approach. "
        "Respond by calling the function 'suggest_numeric_analysis' with a suitable analysis_type."
    )
    messages = [
        {"role": "system", "content": "You are a data scientist."},
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

def save_outputs(base_name, chart_files, readme_content):
    output_dir = base_name
    os.makedirs(output_dir, exist_ok=True)
    for chart in chart_files:
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

    base_name = safe_filename(os.path.basename(csv_file))
    chart_files = []

    corr_file = f"{base_name}_correlation.png"
    if plot_correlation_matrix(df, corr_file):
        chart_files.append(corr_file)

    missing_file = f"{base_name}_missing.png"
    if plot_missing_values(df, missing_file):
        chart_files.append(missing_file)

    hist_file = f"{base_name}_hist.png"
    if plot_hist_of_numeric(df, hist_file):
        chart_files.append(hist_file)

    # LLM analysis
    analysis_insights = get_llm_analysis(summary, cat_analysis, outliers, regression_info)
    run_function_call_scenario(summary)

    # Get narrative story
    readme_content = get_llm_story(summary, analysis_insights, chart_files)

    # Save outputs
    save_outputs(base_name, chart_files, readme_content)

    logging.info("Analysis complete. README.md and charts created.")

if __name__ == "__main__":
    main()
