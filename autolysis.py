# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai>=1.0.0",  # Updated to latest version
#   "tenacity",
#   "chardet",         # Ensure chardet is included for encoding detection
# ]
# ///

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI, APIError 
from tenacity import retry, wait_fixed, stop_after_attempt
import chardet
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ===========================
# Configuration
# ===========================
# Set up the AI Proxy according to the instructions
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

def summarize_dataframe(df, max_unique=10):
    """Create a dictionary summarizing the dataframe's structure and stats."""
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
        # Sample up to max_unique values
        example_vals = unique_vals[:max_unique]
        col_info['example_values'] = [str(v) for v in example_vals]

        # Basic stats if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info['mean'] = float(df[col].mean()) if df[col].count() > 0 else None
            col_info['std'] = float(df[col].std()) if df[col].count() > 1 else None
            col_info['min'] = float(df[col].min()) if df[col].count() > 0 else None
            col_info['max'] = float(df[col].max()) if df[col].count() > 0 else None
        summary['columns'].append(col_info)
    return summary

def plot_correlation_matrix(df, output_file):
    """Plot a correlation matrix of numeric columns."""
    numeric_df = df.select_dtypes(include=['float', 'int'])
    if numeric_df.shape[1] < 2:
        # Not enough numeric columns to plot correlation
        return None
    corr = numeric_df.corr()
    plt.figure(figsize=(6,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', square=True, cbar=True)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    return output_file

def plot_missing_values(df, output_file):
    """Bar chart of missing values per column."""
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
    """Plot a histogram for numeric columns."""
    numeric_df = df.select_dtypes(include=['float','int'])
    if numeric_df.shape[1] == 0:
        return None
    plt.figure(figsize=(6,4))
    # If multiple numeric columns, create a grid of histograms
    # If only one column, a single histogram with KDE.
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
def get_llm_analysis(summary_dict):
    """Ask the LLM to analyze the summary and provide insights."""
    content = (
        "You are a data analyst. I have a dataset summary. "
        "Please provide general insights and suggest deeper analyses.\n\n"
        f"Data summary: {summary_dict}\n"
    )
    messages = [
        {"role": "system", "content": "You are a helpful data analyst."},
        {"role": "user", "content": content}
    ]
    response = call_llm(messages)
    return response.choices[0].message.content.strip()

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def get_llm_story(summary_dict, analysis_insights, chart_files):
    """
    Ask the LLM to write a narrative in Markdown that describes the data,
    the analysis done, insights found, and implications.
    """
    content = (
        "You are a data storyteller. I have a dataset and performed a generic analysis. "
        "I generated charts and derived some insights. Please produce a README.md in Markdown with:\n"
        "- A brief introduction of the dataset\n"
        "- Description of the analyses performed (missing values, correlations, distributions)\n"
        "- The insights discovered from the analysis\n"
        "- Implications or next steps\n"
        "- Include the generated charts using the syntax: ![Chart](filename.png)\n"
        "- Structure the narrative with headings\n\n"
        f"Dataset summary:\n{summary_dict}\n\n"
        f"Insights:\n{analysis_insights}\n\n"
        "Chart files: " + ", ".join(chart_files) + "\n\n"
        "Please produce a coherent and polished narrative."
    )

    messages = [
        {"role": "system", "content": "You are a brilliant data storyteller."},
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
    """Demonstrate function calling scenario to comply with project requirements."""
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

def read_csv_with_encoding(file_path, encodings=['utf-8', 'latin1', 'cp1252']):
    """Attempt to read a CSV file with multiple encodings."""
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            logging.info(f"Successfully read {file_path} with encoding {enc}")
            return df
        except UnicodeDecodeError as e:
            logging.warning(f"Failed to read {file_path} with encoding {enc}: {e}")
    # If all encodings fail, raise an error
    raise UnicodeDecodeError(f"Unable to read {file_path} with tried encodings: {encodings}")

def save_outputs(base_name, chart_files, readme_content):
    """Save charts and README.md into a dedicated directory."""
    output_dir = base_name
    os.makedirs(output_dir, exist_ok=True)
    
    # Move chart files into the directory
    for chart in chart_files:
        os.replace(chart, os.path.join(output_dir, chart))
    
    # Save README.md into the directory
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

    # Use the new function to read the CSV with multiple encoding attempts
    try:
        df = read_csv_with_encoding(csv_file)
    except UnicodeDecodeError as e:
        logging.error(e)
        sys.exit(1)

    # Summarize the dataset
    summary = summarize_dataframe(df)

    # Generate charts
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

    # LLM analysis - get insights
    analysis_insights = get_llm_analysis(summary)

    # Demonstrate function calling scenario
    run_function_call_scenario(summary)

    # Ask the LLM for a narrative story
    readme_content = get_llm_story(summary, analysis_insights, chart_files)

    # Save outputs into a dedicated directory
    save_outputs(base_name, chart_files, readme_content)

    logging.info("Analysis complete. README.md and charts created.")

if __name__ == "__main__":
    main()
