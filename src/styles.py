import streamlit as st

def load_css():
    """Inject custom CSS to style the app with a dark theme."""
    st.markdown(
        """
        <style>
        /* --- Variabel Warna --- */
        :root {
            --color-bg-primary: #111827; /* Main background */
            --color-bg-secondary: #1F2937; /* Sidebar, card background */
            --color-bg-tertiary: #374151; /* Input background, borders */
            --color-text-primary: #F9FAFB; /* Main text */
            --color-text-secondary: #9CA3AF; /* Subtitles, muted text */
            --color-accent: #EF4444; /* Aksen untuk item aktif/penting */
        }

        /* --- General App & Body --- */
        .stApp {
            background-color: var(--color-bg-primary);
            color: var(--color-text-primary);
        }

        /* --- Sidebar Styling --- */
        [data-testid="stSidebar"] {
            background-color: var(--color-bg-secondary);
            border-right: 1px solid var(--color-bg-tertiary);
        }
        [data-testid="stSidebar"] .stMarkdown {
            color: var(--color-text-primary);
        }
        [data-testid="stSidebar"] .stSelectbox, 
        [data-testid="stSidebar"] .stDateInput,
        [data-testid="stSidebar"] .stMultiSelect {
            background-color: var(--color-bg-tertiary);
            border-radius: 0.5rem;
        }

        /* --- Main Content Styling --- */
        .app-header {
            font-size: 28px; 
            font-weight: 700; 
            margin-bottom: 0.2rem;
            color: var(--color-text-primary);
        }
        .app-subtitle {
            color: var(--color-text-secondary); 
            margin-bottom: 1rem;
        }
        
        /* --- Metric Cards --- */
        .metric-card {
            border-radius: 1rem; 
            padding: 1.5rem; 
            background-color: var(--color-bg-secondary); 
            border: 1px solid var(--color-bg-tertiary);
            margin-bottom: 1rem;
        }
        .metric-card-title {
            font-size: 0.9rem;
            color: var(--color-text-secondary);
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        .metric-card [data-testid="stMetricValue"] {
            font-size: 2.5rem;
            color: var(--color-text-primary);
        }
        .metric-card [data-testid="stMetricLabel"] {
            margin-bottom: 0; /* remove default bottom margin */
        }
        .metric-card .small-muted {
            color: var(--color-text-secondary);
            font-size: 0.85rem;
        }
        
        /* --- Dataframe / Table --- */
        .stDataFrame {
            border: 1px solid var(--color-bg-tertiary);
            border-radius: 0.75rem;
            background-color: var(--color-bg-secondary);
        }
        
        /* --- Buttons --- */
        .stButton button {
            background-color: var(--color-bg-tertiary);
            color: var(--color-text-primary);
            border: 1px solid var(--color-bg-tertiary);
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
        }
        .stButton button:hover {
            background-color: #4B5563;
            border-color: #4B5563;
        }

        /* --- Hapus footer bawaan Streamlit --- */
        footer {
            visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

