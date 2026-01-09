import streamlit as st

def render_sidebar():
    st.sidebar.title("Configuration")
    
    config = {}
    config["dataset_dir"] = st.sidebar.text_input("Dataset Directory", "data")
    config["model_name"] = st.sidebar.text_input("Model Name/ID", "MobileCLIP-S2")
    config["pretrained_tag"] = st.sidebar.text_input("Pretrained Tag", "openai")
    config["filter_json_path"] = st.sidebar.text_input("Tag JSON Path", "filter_attributes.json")
    config["text_json_path"] = st.sidebar.text_input("Text Tag JSON Path", "text_attributes.json")
    config["mapping_path"] = st.sidebar.text_input("Mapping JSON Path", "mapping.json")
    
    st.sidebar.markdown("### Auto-labeling Config")
    config["api_url"] = st.sidebar.text_input("API URL", "http://localhost:8000/v1")
    config["api_key"] = st.sidebar.text_input("API Key", "sk-...", type="password")
    config["vlm_model"] = st.sidebar.text_input("VLM Model Name", "gpt-4o-mini")
    
    return config
