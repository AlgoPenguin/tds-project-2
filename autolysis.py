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
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA

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

    Parameters:
        df (pd.DataFrame): The input dataframe.
        max_unique (int): Max number of unique values to sample for display.

    Returns:
        dict: A dictionary summarizing shape, columns, types, missing counts, and basic stats.
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

    Parameters:
        df (pd.DataFrame): The input dataframe.
        max_levels (int): Maximum number of top categories to display.

    Returns:
        list of dict: Each dict has 'column', 'top_values', 'num_unique'.
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

    Returns:
        list of dict: Each dict with 'column', 'outlier_count', 'total_count'.
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

    Returns:
        dict or None: Regression info including coef, r_squared, p-values or None if not possible.
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
    
    Returns:
        dict with PCA results and a plot file if successful, else None.
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=0)
    if numeric_df.shape[1] <= 2 or numeric_df.shape[0] < 5:
        return None

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(numeric_df)
    pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    # Try to color by the first categorical column if available
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != 'datetime64[ns]']
    if cat_cols:
        hue_col = cat_cols[0]
        pca_df[hue_col] = df[hue_col].iloc[pca_df.index].fillna("Missing")
    else:
        hue_col = None

    pca_file = "pca_scatter.png"
    plt.figure(figsize=(6,4))
    if hue_col:
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue=hue_col, palette="Set2")
    else:
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", color='blue')
    plt.title("PCA Scatter Plot (First Two PCs)")
    plt.tight_layout()
    plt.savefig(pca_file)
    plt.close()

    return {
        'explained_variance': pca.explained_variance_ratio_.tolist(),
        'pca_file': pca_file,
        'hue_col': hue_col
    }

def plot_correlation_matrix(df, output_file):
    """
    Plot correlation matrix of numeric columns.
    
    Returns:
        output_file or None if not enough numeric columns.
    """
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
    """
    Plot a bar chart of missing values per column.
    
    Returns:
        output_file or None if no missing values.
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
    
    Returns:
        output_file or None if no numeric data.
    """
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
def get_llm_analysis(summary_dict, cat_analysis, outliers, regression_info, pca_info):
    """
    Use LLM to analyze the summary and additional results.

    Parameters:
        summary_dict (dict): Summary of the dataset.
        cat_analysis (list): Categorical analysis.
        outliers (list): Outlier info.
        regression_info (dict or None): Regression analysis results.
        pca_info (dict or None): PCA results.

    Returns:
        str: Insights from the LLM.
    """
    # If there's PCA info, we mention advanced analysis.
    # Also instruct the LLM to maintain consistent interpretation across runs.
    content = (
        "You are a data analyst. We have a dataset summary and various analyses.\n"
        "Be stable and consistent in your reasoning; do not contradict yourself if run multiple times.\n\n"
        "Data summary (concise):\n" + str(summary_dict) + "\n\n"
        "Categorical analysis:\n" + str(cat_analysis) + "\n\n"
        "Outlier detection:\n" + str(outliers) + "\n\n"
        "Regression analysis:\n" + str(regression_info) + "\n\n"
        "PCA analysis:\n" + str(pca_info) + "\n\n"
        "Discuss numeric and categorical aspects, highlight outliers and their potential implications, interpret the regression, "
        "and consider what PCA results suggest. Mention that results may vary, but aim for stable interpretation. Suggest deeper analyses or advanced statistical tests next time."
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
    Ask the LLM to write a coherent README.md narrative referencing the analyses and charts.

    Parameters:
        summary_dict (dict): Data summary.
        analysis_insights (str): Insights from LLM analysis.
        chart_files (list): List of chart filenames generated.

    Returns:
        str: The narrative in Markdown.
    """
    # Emphasize stable and coherent narrative and direct mention of charts:
    content = (
        "You are a data storyteller. Produce a stable and coherent README.md. "
        "We analyzed a dataset by checking missing values, correlations, distributions, categorical frequencies, outliers, regression, and PCA.\n\n"
        "Data summary (concise):\n" + str(summary_dict) + "\n\n"
        "Insights:\n" + analysis_insights + "\n\n"
        "Charts available: " + ", ".join(chart_files) + "\n\n"
        "Instructions:\n"
        "- Start with an introduction.\n"
        "- Describe each analysis step and why it was performed.\n"
        "- Reference the charts at appropriate steps using `![Chart](filename.png)`.\n"
        "- Summarize key findings from numeric, categorical, outlier, regression, and PCA analyses.\n"
        "- Suggest implications and next steps.\n"
        "- Remind the reader that LLM results might vary, but you've tried to maintain consistency.\n"
    )
    messages = [
        {"role": "system", "content": "You are a brilliant data storyteller. Maintain coherence and consistent interpretation."},
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
    """
    Demonstrate function calling scenario to comply with project requirements.
    """
    content = (
        "Given the data summary, suggest a numeric analysis approach. "
        "Respond by calling the function 'suggest_numeric_analysis' with a suitable analysis_type."
    )
    messages = [
        {"role": "system", "content": "You are a data scientist trying to be consistent in recommendations."},
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
    """
    Save charts and README.md into a dedicated directory.

    Parameters:
        base_name (str): Base name derived from the input CSV.
        chart_files (list): List of generated chart files.
        readme_content (str): The narrative for README.md.
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

    # LLM analysis
    analysis_insights = get_llm_analysis(summary, cat_analysis, outliers, regression_info, pca_info)
    run_function_call_scenario(summary)

    # Get narrative story
    readme_content = get_llm_story(summary, analysis_insights, chart_files)

    # Save outputs
    save_outputs(base_name, chart_files, readme_content)

    logging.info("Analysis complete. README.md and charts created.")

if __name__ == "__main__":
    main()
