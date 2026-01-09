import streamlit as st
import os

DEBUG_ROOT = "debug"
if not os.path.exists(DEBUG_ROOT):
    os.makedirs(DEBUG_ROOT)

def render_debug_view_tab():
    st.header("Debug View")

    # List all directories in debug root
    debug_runs = sorted([d for d in os.listdir(DEBUG_ROOT) if os.path.isdir(os.path.join(DEBUG_ROOT, d))], reverse=True)

    selected_debug_run = st.selectbox("Select Debug Run", [""] + debug_runs)

    if selected_debug_run:
        run_path = os.path.join(DEBUG_ROOT, selected_debug_run)
        # Structure: run_path / class_folder / images...

        classes = sorted([d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d))])

        selected_class = st.selectbox("Select Class/Query", [""] + classes)

        if selected_class:
            class_path = os.path.join(run_path, selected_class)
            files = sorted([f for f in os.listdir(class_path) if not f.endswith(".txt")])

            st.write(f"Found {len(files)} cases.")

            # Pagination or grid
            for f in files:
                col1, col2 = st.columns([1, 2])
                img_p = os.path.join(class_path, f)
                txt_p = img_p + ".debug.txt"

                with col1:
                    st.image(img_p, use_container_width=True)
                with col2:
                    if os.path.exists(txt_p):
                        with open(txt_p, "r") as tf:
                            content = tf.read()
                        st.text_area(f"Details for {f}", content, height=200)
                    else:
                        st.write("No debug info file.")
                st.divider()
