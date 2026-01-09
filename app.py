import streamlit as st
import os

from src.ui.sidebar import render_sidebar
from src.ui.tabs.run_evaluation import render_run_evaluation_tab
from src.ui.tabs.debug_view import render_debug_view_tab
from src.ui.tabs.auto_labeling import render_auto_labeling_tab
from src.ui.tabs.history_compare import render_history_compare_tab
from src.ui.tabs.dataset_analysis import render_dataset_analysis_tab

# --- Config ---
st.set_page_config(page_title="CLIP Eval Tool", layout="wide")

# --- Sidebar ---
config = render_sidebar()

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Run Evaluation", "Debug View", "Auto-labeling", "History & Compare", "Dataset Analysis"])

with tab1:
    render_run_evaluation_tab(config)

with tab2:
    render_debug_view_tab()

with tab3:
    render_auto_labeling_tab(config)

with tab4:
    render_history_compare_tab(config)

with tab5:
    render_dataset_analysis_tab(config)
