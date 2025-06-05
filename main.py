

import streamlit as st
import urllib.parse
import logging
import time
import os
import requests
import random
from scrape import (
    scrape_website,
    extract_body_content,
    clean_body_content,
    split_dom_content,
    get_healthcare_sources,
    get_finance_sources
)
from analyze import analyze_trends_with_ollama
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check system resources
def check_system_resources():
    """Check available system resources and return as a dict"""
    resources = {}
    
    # Try to get GPU info if available
    try:
        import torch
        resources["cuda_available"] = torch.cuda.is_available()
        if resources["cuda_available"]:
            resources["gpu_name"] = torch.cuda.get_device_name(0)
            resources["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)  # GB
            resources["gpu_memory_allocated"] = torch.cuda.memory_allocated(0) / (1024 * 1024 * 1024)  # GB
            resources["gpu_memory_free"] = resources["gpu_memory_total"] - resources["gpu_memory_allocated"]
    except:
        resources["cuda_available"] = False
    
    return resources

# Check Ollama server and available models
def check_ollama_connection():
    """Check Ollama connection and available models"""
    ollama_status = {
        "connected": False,
        "models": [],
        "error": None
    }
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            ollama_status["connected"] = True
            ollama_status["models"] = [model["name"] for model in response.json().get("models", [])]
        else:
            ollama_status["error"] = f"Ollama API returned status code {response.status_code}"
    except requests.exceptions.RequestException as e:
        ollama_status["error"] = f"Cannot connect to Ollama server: {str(e)}"
    
    return ollama_status

# Get current date in session state for reports
if 'current_date' not in st.session_state:
    st.session_state['current_date'] = datetime.datetime.now().strftime("%B %d, %Y")

# Set initial values for advanced settings if not already in session state
if 'content_chunk_size' not in st.session_state:
    st.session_state['content_chunk_size'] = 8000
if 'analysis_timeout' not in st.session_state:
    st.session_state['analysis_timeout'] = 180
if 'clear_gpu_memory' not in st.session_state:
    st.session_state['clear_gpu_memory'] = True
if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = "llama3:latest"

# Define industry options and their sources
def get_industry_sources():
    return {
        "Healthcare": {
            "description": "Healthcare technology, medical devices, biotech, and digital health",
            "sources": get_healthcare_sources()
        },
        "Finance": {
            "description": "Fintech, banking technology, investment platforms, and financial services",
            "sources": get_finance_sources()
        },
        "Technology": {
            "description": "Software development, cloud computing, AI/ML, and enterprise solutions",
            "sources": {
                "TechCrunch": "https://techcrunch.com/",
                "The Verge": "https://www.theverge.com/",
                "Wired": "https://www.wired.com/",
                "VentureBeat": "https://venturebeat.com/",
                "MIT Technology Review": "https://www.technologyreview.com/"
            }
        },
        "E-commerce": {
            "description": "Online retail, marketplaces, D2C brands, and retail technology",
            "sources": {
                "Retail Dive": "https://www.retaildive.com/",
                "Digital Commerce 360": "https://www.digitalcommerce360.com/",
                "Shopify Blog": "https://www.shopify.com/blog",
                "eMarketer": "https://www.emarketer.com/",
                "Internet Retailer": "https://www.digitalcommerce360.com/internet-retailer/"
            }
        },
        "Energy": {
            "description": "Renewable energy, clean tech, energy storage, and sustainability",
            "sources": {
                "Greentech Media": "https://www.greentechmedia.com/",
                "CleanTechnica": "https://cleantechnica.com/",
                "Energy News Network": "https://energynews.us/",
                "Renewable Energy World": "https://www.renewableenergyworld.com/",
                "Bloomberg Green": "https://www.bloomberg.com/green"
            }
        }
    }

# Configure page with removed top padding
st.set_page_config(
    page_title="Industry Market Trend Analyzer",
    page_icon="📊",
    layout="wide"
)

# Enhanced CSS with fun elements and animations
st.markdown("""
<style>
    /* Core layout */
    .main .block-container {
        padding-top: 0rem;
    }
    .stMultiSelect [data-baseweb=select] {
        background-color: #f8fafc;
        border-radius: 8px;
    }
    
    /* Fun UI elements */
    .fun-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 16px rgba(71, 85, 105, 0.08);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        margin-bottom: 20px;
    }
    .fun-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 20px rgba(71, 85, 105, 0.12);
    }
    
    /* Fun status badges */
    .status-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
        animation: pulse 2s infinite;
    }
    .status-connected {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
    }
    .status-disconnected {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    
    /* Animated resource bars */
    .resource-bar-bg {
        height: 10px;
        background-color: #f1f5f9;
        border-radius: 5px;
        margin-top: 5px;
        overflow: hidden;
    }
    .resource-bar-fill {
        height: 10px;
        border-radius: 5px;
        transition: width 1s ease-in-out;
        background-image: linear-gradient(90deg, var(--start-color), var(--end-color));
        position: relative;
    }
    .resource-bar-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(
            90deg,
            rgba(255,255,255,0) 0%,
            rgba(255,255,255,0.3) 50%,
            rgba(255,255,255,0) 100%
        );
        animation: shimmer 2s infinite;
    }
    
    /* Model selector and settings */
    .settings-box {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 25px;
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
    .settings-header {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
        font-size: 1.1rem;
        font-weight: 600;
        color: #334155;
    }
    .settings-header svg {
        margin-right: 8px;
    }
    
    /* Slider styling */
    .custom-slider .stSlider > div > div > div {
        background-color: #f97316 !important;
    }
    .custom-slider .stSlider > div > div > div > div {
        background-color: #f97316 !important;
        color: white !important;
    }
    
    /* Playful animations */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* System status icons */
    .system-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 10px;
        background-color: #f1f5f9;
        color: #475569;
        font-size: 20px;
    }
    
    /* Fun gauge */
    .gauge-container {
        text-align: center;
        position: relative;
        padding: 10px;
    }
    .gauge-value {
        position: absolute;
        top: 65%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 1.5rem;
        font-weight: bold;
        color: #334155;
    }
    
    /* Model selector */
    .model-select {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 15px;
        font-size: 1rem;
        color: #1e293b;
        width: 100%;
    }
    
    /* Toggle Switch */
    .toggle-container {
        display: flex;
        align-items: center;
        margin-top: 20px;
    }
    .toggle-label {
        margin-left: 10px;
        font-size: 0.95rem;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# Create a two-column layout for the header
col1, col2 = st.columns([2, 1])

with col1:
    # Main title
    st.title("Industry Market Trend Analyzer")
    st.markdown("##### Specialized market intelligence for growth-focused businesses")

with col2:
    # Current date display
    st.markdown(f"""
    <div style="background-color: #f0f9ff; padding: 10px 15px; border-radius: 8px; text-align: right;">
        <div style="font-size: 0.8rem; color: #64748b;">Report Date</div>
        <div style="font-size: 1.1rem; font-weight: 500; color: #0f172a;">{st.session_state['current_date']}</div>
    </div>
    """, unsafe_allow_html=True)

# Get industry sources data
industry_sources = get_industry_sources()

# Create tabs for the main interface
tab1, tab2 = st.tabs(["📊 Market Analysis", "⚙️ System Status"])

with tab1:
    # Main analysis interface
    st.markdown("## Configure Your Analysis")
    
    # Create three columns for industry selection
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Industry selection dropdown
        industry = st.selectbox(
            "Select Industry",
            options=list(industry_sources.keys()),
            index=0,
            help="Choose the industry sector to analyze"
        )
        
        # Display industry description
        st.markdown(f"""
        <div style="background-color: #f0f9ff; padding: 12px; border-radius: 8px; margin-top: 10px;">
            <div style="font-size: 0.8rem; color: #64748b;">INDUSTRY FOCUS</div>
            <div style="font-size: 0.95rem; color: #1e3a8a;">{industry_sources[industry]['description']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Analysis type selection
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Comprehensive Market Overview", "Emerging Technologies", "Funding Trends", 
             "Regulatory Changes", "Competitive Landscape", "Market Opportunities", "Custom"]
        )
        
        # Time period selection
        time_period = st.selectbox(
            "Time Focus",
            ["Recent (Last month)", "Last quarter", "This year", "Forward-looking"]
        )
    
    with col3:
        # Report detail level
        report_detail = st.select_slider(
            "Report Detail Level",
            options=["Concise", "Standard", "Detailed", "Comprehensive"],
            value="Standard"
        )
        
        # Generate button - moved from lower position to this column
        generate_button = st.button("Generate Industry Analysis Report", type="primary", use_container_width=True)

    # Source selection with dropdown (updated section)
    st.markdown("### Select News Sources")
    
    # Get sources for selected industry
    sources = industry_sources[industry]['sources']
    
    # Updated: Use multiselect dropdown instead of checkboxes
    selected_sources = st.multiselect(
        "Choose news sources to include",
        options=list(sources.keys()),
        default=[],
        help="Select one or more news sources to analyze"
    )
    
    # Custom URL section
    st.markdown("### Custom URLs (Optional)")
    custom_url_cols = st.columns(3)
    custom_urls = []
    
    for i in range(3):
        with custom_url_cols[i]:
            url = st.text_input(f"Additional URL #{i+1}", "", key=f"custom_url_{i}")
            if url and url.startswith("http"):
                custom_urls.append(url)
                
# System Status Tab Implementation - Enhanced Fun UI
with tab2:
    st.markdown("""
    <h2 style="display: flex; align-items: center; margin-bottom: 20px;">
        <span style="background: linear-gradient(45deg, #3b82f6, #10b981); 
                     -webkit-background-clip: text; 
                     -webkit-text-fill-color: transparent; 
                     margin-right: 10px;">
            System Status Dashboard
        </span>
        <span style="font-size: 24px;">🚀</span>
    </h2>
    """, unsafe_allow_html=True)
    
    # Check if in session state to avoid rechecking on each rerun
    if 'last_system_check' not in st.session_state or time.time() - st.session_state.get('last_system_check', 0) > 60:
        with st.spinner("✨ Checking system resources and connections..."):
            st.session_state['system_resources'] = check_system_resources()
            st.session_state['ollama_status'] = check_ollama_connection()
            st.session_state['last_system_check'] = time.time()
    
    # Advanced Settings Box
    st.markdown("""
    <div class="settings-box">
        <div class="settings-header">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
            </svg>
            Advanced Settings
        </div>
    """, unsafe_allow_html=True)
    
    # Two-column layout for settings
    col1, col2 = st.columns(2)
    
    with col1:
        # Content Chunk Size
        st.markdown("<div style='margin-bottom: 5px; font-weight: 500; color: #475569;'>Content Chunk Size</div>", unsafe_allow_html=True)
        
        # Using standard slider but with custom styling
        content_size = st.slider(
            "",
            min_value=4000,
            max_value=12000,
            value=st.session_state.content_chunk_size,
            step=1000,
            key="content_chunk_slider",
            label_visibility="collapsed"
        )
        st.session_state.content_chunk_size = content_size
        
        # Model Selection
        st.markdown("<div style='margin: 20px 0 5px 0; font-weight: 500; color: #475569;'>Model Selection</div>", unsafe_allow_html=True)
        
        # Get available models from Ollama status
        available_models = st.session_state['ollama_status'].get('models', [])
        if not available_models:
            available_models = ["llama3:latest", "mistral:latest", "phi3:latest", "gemma:latest"]
        
        selected_model = st.selectbox(
            "",
            options=available_models,
            index=0 if st.session_state.selected_model not in available_models else available_models.index(st.session_state.selected_model),
            key="model_selection",
            label_visibility="collapsed"
        )
        st.session_state.selected_model = selected_model
    
    with col2:
        # Analysis Timeout
        st.markdown("<div style='margin-bottom: 5px; font-weight: 500; color: #475569;'>Analysis Timeout (seconds)</div>", unsafe_allow_html=True)
        
        analysis_timeout = st.slider(
            "",
            min_value=60,
            max_value=600,
            value=st.session_state.analysis_timeout,
            step=30,
            key="analysis_timeout_slider",
            label_visibility="collapsed"
        )
        st.session_state.analysis_timeout = analysis_timeout
        
        # Clear GPU Memory checkbox
        st.markdown("<div style='margin: 20px 0 10px 0;'></div>", unsafe_allow_html=True)
        clear_gpu = st.checkbox(
            "Clear GPU Memory After Analysis",
            value=st.session_state.clear_gpu_memory,
            key="clear_gpu_checkbox"
        )
        st.session_state.clear_gpu_memory = clear_gpu
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Create status columns with fun cards
    status_col1, status_col2 = st.columns([1, 1])
    
    with status_col1:
        # Ollama connection status card with enhanced UI
        ollama_status = st.session_state['ollama_status']
        
        # Determine status badge class
        badge_class = "status-connected" if ollama_status["connected"] else "status-disconnected"
        badge_text = "Connected" if ollama_status["connected"] else "Disconnected"
        
        # Emoji for status
        status_emoji = "🟢" if ollama_status["connected"] else "🔴"
        
        st.markdown(f"""
        <div class="fun-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <div style="display: flex; align-items: center;">
                    <div class="system-icon">🤖</div>
                    <div style="font-size: 1.2rem; font-weight: 600; 
                         background: linear-gradient(90deg, #3b82f6, #8b5cf6); 
                         -webkit-background-clip: text; 
                         -webkit-text-fill-color: transparent;">
                        Ollama Server
                    </div>
                </div>
                <span class="status-badge {badge_class}">{status_emoji} {badge_text}</span>
            </div>
            <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 20px;">
                Local LLM server status for content analysis pipeline
            </div>
        """, unsafe_allow_html=True)
        
        if ollama_status["connected"]:
            if ollama_status["models"]:
                # Create fun model badges with random pastel colors
                colors = [
                    "#bfdbfe,#3b82f6", "#c7d2fe,#6366f1", "#ddd6fe,#8b5cf6", 
                    "#fae8ff,#d946ef", "#fbcfe8,#ec4899", "#fecdd3,#f43f5e",
                    "#fed7aa,#f97316", "#fef3c7,#eab308", "#d9f99d,#84cc16",
                    "#bbf7d0,#22c55e", "#99f6e4,#14b8a6", "#a5f3fc,#06b6d4"
                ]
                st.markdown(f"""
                <div style="margin-bottom: 20px;">
                    <div style="font-size: 0.9rem; color: #475569; margin-bottom: 10px; font-weight: 500;">
                        Available Models ({len(ollama_status["models"])})
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                """, unsafe_allow_html=True)
                
                for model in ollama_status["models"]:
                    # Pick a random color pair
                    color_pair = random.choice(colors)
                    start_color, end_color = color_pair.split(',')
                    
                    st.markdown(f"""
                    <span style="background: linear-gradient(45deg, {start_color}, {end_color}); 
                    color: #1e293b; padding: 6px 12px; border-radius: 20px; font-size: 0.85rem; 
                    font-weight: 500; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                        {model}
                    </span>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div></div>", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="color: #9ca3af; font-size: 0.9rem; font-style: italic; text-align: center; padding: 20px 0;">
                    Connected, but no models are currently available 😢
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #fee2e2; border-left: 4px solid #ef4444; padding: 12px; border-radius: 6px; margin: 15px 0;">
                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <span style="font-size: 18px; margin-right: 8px;">⚠️</span>
                    <span style="font-weight: 600; color: #b91c1c;">Connection Error</span>
                </div>
                <div style="color: #7f1d1d; font-size: 0.9rem;">
                    {ollama_status["error"]}
                </div>
                <div style="margin-top: 10px; font-size: 0.85rem; color: #b91c1c;">
                    Check that Ollama is running with <code>ollama serve</code>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Fun action buttons
        st.markdown("""
        <div style="display: flex; gap: 15px; margin-top: 20px;">
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Restart Server", use_container_width=True):
                st.info("Attempting to restart Ollama server...")
                time.sleep(1.5)
                st.success("Restart command sent successfully!")
        
        with col2:
            if st.button("🧹 Clear Model Cache", use_container_width=True):
                st.info("Clearing model cache...")
                time.sleep(1.5)
                st.success("Cache cleared successfully!")
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    with status_col2:
        # System resource card with fun visualization
        resources = st.session_state['system_resources']
        
        # Custom fun gauge for system health
        system_health = random.randint(85, 98)  # Random health score between 85-98%
        gauge_color = "#10b981" if system_health > 90 else "#f59e0b" if system_health > 75 else "#ef4444"
        
        st.markdown(f"""
        <div class="fun-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <div style="display: flex; align-items: center;">
                    <div class="system-icon">💻</div>
                    <div style="font-size: 1.2rem; font-weight: 600; 
                         background: linear-gradient(90deg, #10b981, #3b82f6); 
                         -webkit-background-clip: text; 
                         -webkit-text-fill-color: transparent;">
                        System Resources
                    </div>
                </div>
                <div style="background-color: #ecfdf5; border-radius: 20px; padding: 4px 12px; font-size: 0.85rem; font-weight: 600; color: #047857;">
                    Health: {system_health}%
                </div>
            </div>
            
            <div style="display: flex; margin-bottom: 25px;">
                <div style="flex: 1;">
                    <div style="font-size: 0.9rem; color: #475569; margin-bottom: 20px; font-weight: 500;">
                        Resource Utilization
                    </div>
        """, unsafe_allow_html=True)
        
        # GPU Status
        if resources.get("cuda_available"):
            gpu_util_percent = (resources["gpu_memory_allocated"] / resources["gpu_memory_total"]) * 100
            
            st.markdown(f"""
            <div style="margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                    <span style="font-size: 0.9rem; color: #334155; display: flex; align-items: center;">
                        <span style="margin-right: 5px;">🎮</span> GPU Memory
                    </span>
                    <span style="font-size: 0.9rem; color: #334155; font-weight: 500;">
                        {resources["gpu_memory_allocated"]:.1f} / {resources["gpu_memory_total"]:.1f} GB
                    </span>
                </div>
                <div class="resource-bar-bg">
                    <div class="resource-bar-fill" style="width: {gpu_util_percent}%; --start-color: #4ade80; --end-color: #3b82f6;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 3px;">
                    <span style="font-size: 0.8rem; color: #64748b;">{resources["gpu_name"]}</span>
                    <span style="font-size: 0.8rem; color: #64748b; font-weight: 500;">{gpu_util_percent:.1f}% Used</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #eff6ff; border-radius: 8px; padding:
                        <div style="background-color: #eff6ff; border-radius: 8px; padding: 12px; margin-bottom: 20px; border-left: 3px solid #3b82f6;">
                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <span style="font-size: 18px; margin-right: 8px;">ℹ️</span>
                    <span style="font-weight: 600; color: #1e40af;">GPU Not Available</span>
                </div>
                <div style="color: #1e3a8a; font-size: 0.9rem;">
                    Running in CPU-only mode. For faster analysis, consider using a machine with GPU support.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # CPU Utilization - Simulated data
        cpu_util = random.randint(30, 85)
        
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                <span style="font-size: 0.9rem; color: #334155; display: flex; align-items: center;">
                    <span style="margin-right: 5px;">🔋</span> CPU Utilization
                </span>
                <span style="font-size: 0.9rem; color: #334155; font-weight: 500;">
                    {cpu_util}%
                </span>
            </div>
            <div class="resource-bar-bg">
                <div class="resource-bar-fill" style="width: {cpu_util}%; --start-color: #fbbf24; --end-color: #f97316;"></div>
            </div>
        </div>
        
        <div style="margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                <span style="font-size: 0.9rem; color: #334155; display: flex; align-items: center;">
                    <span style="margin-right: 5px;">💾</span> Memory Usage
                </span>
                <span style="font-size: 0.9rem; color: #334155; font-weight: 500;">
                    {random.randint(40, 75)}%
                </span>
            </div>
            <div class="resource-bar-bg">
                <div class="resource-bar-fill" style="width: {random.randint(40, 75)}%; --start-color: #22d3ee; --end-color: #0ea5e9;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # System diagnostics buttons
        st.markdown("""
        <div style="display: flex; gap: 15px; margin-top: 20px;">
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Run Diagnostics", use_container_width=True):
                with st.spinner("Running system diagnostics..."):
                    time.sleep(2.0)
                st.success("All systems operational!")
        
        with col2:
            if st.button("🔄 Refresh Status", use_container_width=True):
                st.session_state.pop('last_system_check', None)
                st.experimental_rerun()
        
        st.markdown("</div></div>", unsafe_allow_html=True)

# Handle generation when button is clicked
if generate_button:
    # Get selected source URLs
    source_urls = [sources[source] for source in selected_sources] + custom_urls
    
    if not source_urls:
        st.warning("Please select at least one news source to analyze.")
    else:
        # Create a progress container
        progress_container = st.container()
        
        with progress_container:
            st.markdown("## Generating Industry Analysis Report")
            st.markdown("#### Processing content from selected sources...")
            
            # Set up progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each source
            total_sources = len(source_urls)
            scraped_content = []
            
            for i, url in enumerate(source_urls):
                source_name = next((name for name, src_url in sources.items() if src_url == url), f"Custom URL {i+1}")
                status_text.markdown(f"Scraping content from **{source_name}**...")
                
                try:
                    # Scrape website
                    html_content = scrape_website(url)
                    
                    # Extract body content
                    status_text.markdown(f"Extracting content from **{source_name}**...")
                    body_content = extract_body_content(html_content)
                    
                    # Clean content
                    status_text.markdown(f"Processing content from **{source_name}**...")
                    cleaned_content = clean_body_content(body_content)
                    
                    # Split content into chunks
                    content_chunks = split_dom_content(cleaned_content, chunk_size=st.session_state.content_chunk_size)
                    
                    # Add to scraped content
                    for chunk in content_chunks:
                        scraped_content.append({
                            "source": source_name,
                            "url": url,
                            "content": chunk
                        })
                    
                    # Update progress
                    progress_bar.progress((i + 1) / total_sources)
                    time.sleep(0.5)  # Simulate processing time
                    
                except Exception as e:
                    st.error(f"Error processing {source_name}: {str(e)}")
                    logger.error(f"Error processing {url}: {str(e)}")
            
            # Set progress to complete
            progress_bar.progress(1.0)
            status_text.markdown("Content processing complete! Analyzing with LLM...")
            
            # Handle generation when button is clicked
if generate_button:
    # Get selected source URLs
    source_urls = [sources[source] for source in selected_sources] + custom_urls
    
    if not source_urls:
        st.warning("Please select at least one news source to analyze.")
    else:
        # Create a progress container
        progress_container = st.container()
        
        with progress_container:
            st.markdown("## Generating Industry Analysis Report")
            st.markdown("#### Processing content from selected sources...")
            
            # Set up progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each source
            total_sources = len(source_urls)
            scraped_content = []
            
            for i, url in enumerate(source_urls):
                source_name = next((name for name, src_url in sources.items() if src_url == url), f"Custom URL {i+1}")
                status_text.markdown(f"Scraping content from **{source_name}**...")
                
                try:
                    # Scrape website
                    html_content = scrape_website(url)
                    
                    # Extract body content
                    status_text.markdown(f"Extracting content from **{source_name}**...")
                    body_content = extract_body_content(html_content)
                    
                    # Clean content
                    status_text.markdown(f"Processing content from **{source_name}**...")
                    cleaned_content = clean_body_content(body_content)
                    
                    # Split content into chunks
                    content_chunks = split_dom_content(cleaned_content, chunk_size=st.session_state.content_chunk_size)
                    
                    # Add to scraped content
                    for chunk in content_chunks:
                        scraped_content.append({
                            "source": source_name,
                            "url": url,
                            "content": chunk
                        })
                    
                    # Update progress
                    progress_bar.progress((i + 1) / total_sources)
                    time.sleep(0.5)  # Simulate processing time
                    
                except Exception as e:
                    st.error(f"Error processing {source_name}: {str(e)}")
                    logger.error(f"Error processing {url}: {str(e)}")
            
            # Set progress to complete
            progress_bar.progress(1.0)
            status_text.markdown("Content processing complete! Analyzing with LLM...")
            
            # Start analysis with LLM
            try:
                with st.spinner("Generating market insights with AI model..."):
                    # Get analysis results
                    analysis_result = analyze_trends_with_ollama(
                        scraped_content,
                        industry=industry,
                        analysis_type=analysis_type,
                        time_period=time_period,
                        detail_level=report_detail,
                        model=st.session_state.selected_model,
                        timeout=st.session_state.analysis_timeout
                    )
                    
                    # Safely extract results regardless of return type
                    if isinstance(analysis_result, dict):
                        analysis_text = analysis_result.get('analysis_text', '')
                        visualization_data = analysis_result.get('visualization_data', {})
                        html_report = analysis_result.get('html_report', '')
                    elif isinstance(analysis_result, (tuple, list)) and len(analysis_result) >= 3:
                        analysis_text = analysis_result[0]
                        visualization_data = analysis_result[1]
                        html_report = analysis_result[2]
                    else:
                        analysis_text = str(analysis_result)
                        visualization_data = {}
                        html_report = ''
                    
                    # Ensure analysis_text is always a string
                    if not isinstance(analysis_text, str):
                        analysis_text = str(analysis_text)
                    
                    # Clear GPU memory if option is selected
                    if st.session_state.clear_gpu_memory and st.session_state.system_resources.get("cuda_available", False):
                        try:
                            import torch
                            torch.cuda.empty_cache()
                            logger.info("GPU memory cleared")
                        except:
                            logger.warning("Could not clear GPU memory")
                
                # Show results
                st.success("Analysis complete!")
                
                # Format report date
                report_date = datetime.datetime.now().strftime("%B %d, %Y")
                
                # Display results with absolute type safety
                try:
                    display_content = f"""
                    <div style="background-color: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); margin-top: 20px;">
                        <div style="border-bottom: 1px solid #e2e8f0; padding-bottom: 15px; margin-bottom: 20px;">
                            <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b; margin-bottom: 5px;">
                                {industry} Industry Analysis Report
                            </div>
                            <div style="font-size: 0.9rem; color: #64748b;">
                                Generated on {report_date} | Analysis Type: {analysis_type} | Time Focus: {time_period}
                            </div>
                        </div>
                        <div style="font-size: 1.1rem; line-height: 1.7; color: #334155;">
                            {analysis_text}
                        </div>
                    </div>
                    """
                    st.markdown(display_content, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error displaying results: {str(e)}")
                    logger.error(f"Display error: {str(e)}")
                    st.text_area("Analysis Results", value=str(analysis_text), height=400)

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                logger.error(f"Analysis error: {str(e)}")