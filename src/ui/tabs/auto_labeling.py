import streamlit as st
import sys
import subprocess

def render_auto_labeling_tab(config):
    st.header("Auto-labeling")
    st.write("This tool will label images with attributes: Person Size, Time of Day, Blurry, Resolution.")

    if st.button("Start Auto-labeling"):
        cmd = [
            sys.executable, "src/autolabel.py",
            "--dataset", config["dataset_dir"],
            "--output", config["filter_json_path"],
            "--api_url", config["api_url"],
            "--api_key", config["api_key"],
            "--model", config["vlm_model"]
        ]

        with st.spinner("Labeling... check terminal for progress logs."):
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                st.success("Auto-labeling complete!")
                st.text_area("Output", stdout)
            else:
                st.error("Auto-labeling failed.")
                st.text_area("Error", stderr)

    st.divider()
    st.markdown("### Text Auto-labeling")
    if st.button("Start Text Auto-labeling"):
        cmd = [
            sys.executable, "src/autolabel_text.py",
            "--dataset", config["dataset_dir"],
            "--output", config["text_json_path"],
            "--api_url", config["api_url"],
            "--api_key", config["api_key"],
            "--model", config["vlm_model"]
        ]

        with st.spinner("Labeling Text... check terminal for progress logs."):
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                st.success("Text Auto-labeling complete!")
                st.text_area("Output", stdout)
            else:
                st.error("Text Auto-labeling failed.")
                st.text_area("Error", stderr)
