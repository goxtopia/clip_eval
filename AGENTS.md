# CLIP Eval Tool - Current Status & Instructions

## Project Overview
This project is a CLIP Evaluation Tool designed to assess the performance of CLIP-like models (e.g., MobileCLIP) on custom datasets. It provides a Streamlit-based web interface for running evaluations, automatically labeling dataset attributes, and comparing historical run results.

## Key Features

### 1. Evaluation Interface (`app.py` - Tab 1)
-   **Data Loading:** Loads images and text labels from a specified directory. Uses a mapping JSON to link images to their ground truth text.
-   **Filtering:** Allows users to filter the dataset based on pre-defined attributes (e.g., "Time: Day", "Person Size: Large") stored in a filter JSON file.
-   **Metrics:** Calculates **Global Top-1 Accuracy** for Image-to-Text (I2T) retrieval.
-   **Matrix Analysis:** Breaks down accuracy by attribute values (e.g., accuracy for "Day" vs. "Night"). Includes a default "All" view and optional "Support (Count)" overlay.
-   **History Saving:** Automatically saves every run's results to `history/run_{timestamp}.json`.
-   **Note on T2I:** While the backend (`src/metrics.py`) contains logic for Text-to-Image (T2I) metrics, the current UI primarily exposes and runs Image-to-Text (I2T) evaluation.

### 2. Auto-labeling (`src/autolabel.py` - Tab 2)
-   **Function:** Automatically extracts attributes from images to populate the filter JSON.
-   **Person Size:** Uses **YOLO11x** (`ultralytics`) to detect people and categorize their size relative to the image area (Small, Medium, Large, None).
-   **Visual Attributes:** Uses an **OpenAI-compatible VLM API** (e.g., GPT-4o) to determine:
    -   `time_of_day`: Day / Night
    -   `is_blurry`: Yes / No
    -   `resolution`: High / Low
-   **Optimization:** Resizes images to a maximum dimension of 1024px before sending to the VLM to reduce latency and cost.
-   **Storage:** Attributes are stored keyed by the image's MD5 hash to avoid re-processing duplicates.

### 3. History & Comparison (`app.py` - Tab 4)
-   **Comparison:** Allows users to select multiple past runs from the `history/` folder and choose a **Baseline Run**.
-   **Reporting:** Generates a unified HTML report containing:
    -   A summary table of runs (Model, Samples, Top-1 Acc, Active Filters).
    -   Detailed **Matrix Breakdown** tables comparing accuracy across attributes for each run.
    -   **Comparison Heatmaps (Delta):** Visualizes the accuracy difference (Run - Baseline) for "All Tags" interactions.
        -   **Yellow:** Negative difference (Regression).
        -   **Blue:** Positive difference (Improvement).
    -   Downloadable HTML report.
-   **Visualization:** Displays a simple bar chart of Global Top-1 accuracy for selected runs.

### 4. Dataset Analysis (`app.py` - Tab 5)
-   **Function:** Analyzes and visualizes the distribution of attributes/tags within the dataset.
-   **Features:**
    -   Loads the dataset and filter JSON.
    -   Aggregates counts for every attribute key-value pair.
    -   Displays frequency tables and bar charts for each attribute category (e.g., Person Size distribution, Time of Day distribution).

## Project Structure
-   `app.py`: Main Streamlit entry point. Handles UI, evaluation orchestration, and reporting.
-   `src/autolabel.py`: Script for auto-labeling images. Can be run via UI or CLI.
-   `src/data.py`: Handles data loading and dataset management.
-   `src/model.py`: Wraps the CLIP model interactions (loading, encoding). Supports both `open_clip` and Hugging Face `transformers` backends. Includes caching and "Red vs Blue" sanity check.
-   `src/metrics.py`: Computes I2T metrics (Top-1, Top-5, per-class).
-   `history/`: Directory where evaluation results are saved.

## How to Operate

### Running the App
```bash
streamlit run app.py
```

### Configuration
-   **Sidebar:** Set paths for dataset, filter JSON, and mapping JSON. Configure VLM API details for auto-labeling.

### Workflow
1.  **Auto-labeling (Optional but Recommended):** Go to Tab 2, configure API settings, and click "Start Auto-labeling" to generate attributes for your dataset.
2.  **Run Evaluation:** Go to Tab 1, load data, apply desired filters, and click "Run". Results will be displayed and saved.
3.  **Compare Results:** Go to Tab 3, select previous runs, and click "Generate Comparison Report" to analyze performance changes.

## Environment Dependencies
-   `streamlit`
-   `torch`, `torchvision`
-   `open_clip_torch`
-   `ultralytics` (for YOLO)
-   `openai` (for VLM)
-   `opencv-python`
-   `pandas`
-   `seaborn`, `matplotlib`
