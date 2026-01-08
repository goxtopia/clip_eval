# CLIP Eval Tool

A Streamlit-based tool for evaluating CLIP-like models on custom datasets. This tool provides features for dataset auto-labeling, detailed metric analysis (I2T), and historical run comparison.

## Features

-   **Interactive Evaluation:** Run Image-to-Text retrieval evaluations on your own datasets.
-   **Dataset Auto-labeling:** Automatically generate metadata (Person Size, Time of Day, etc.) using YOLO and VLM APIs (e.g., GPT-4o).
-   **Detailed Analysis:** View global Top-1 accuracy and drill down into performance by attribute (e.g., "How well does the model perform on 'Night' images vs 'Day' images?").
-   **History & Comparison:** Save run results automatically and compare multiple model runs with generated HTML reports.
-   **Flexible Backend:** Supports `open_clip` models and Hugging Face `transformers` models.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Start the App
```bash
streamlit run app.py
```

### 2. Prepare Data
The tool expects a dataset directory containing images (`.jpg`) and corresponding text files (`.txt`).
-   `data/image1.jpg`
-   `data/image1.txt` (contains ground truth labels, one per line)

### 3. Workflow
-   **Auto-labeling (Tab 2):**
    -   Enter your OpenAI-compatible API details.
    -   Run the auto-labeler to generate a filter JSON file (e.g., `filter_attributes.json`).
-   **Run Evaluation (Tab 1):**
    -   Load your dataset.
    -   Select filters if desired (e.g., test only on "Large Person" images).
    -   Click "Run" to evaluate the model.
-   **Compare (Tab 3):**
    -   Select past runs to generate a comparison report.

## Project Structure

-   `app.py`: Main application interface.
-   `src/`: Core logic for data loading, model inference, and metrics.
-   `history/`: Stores JSON results of evaluation runs.
-   `AGENTS.md`: Detailed developer documentation and project status.

## License
[Insert License Here]
