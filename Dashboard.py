import streamlit as st
import pandas as pd
import duckdb as duck
import plotly.express as px
from plotly.subplots import make_subplots
from umap.umap_ import UMAP
from hdbscan import HDBSCAN

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title='Clustering on Countries', layout='wide')
st.title('Clustering on Countries 🌍')

# -----------------------------
# Safe DB Connection
# -----------------------------
import pathlib

BASE_DIR = pathlib.Path(__file__).parent
db_path = BASE_DIR / "country_cluster.db"
conn = duck.connect(str(db_path))

# -----------------------------
# Problem Description
# -----------------------------
st.markdown("""
# Problem Statement

HELP International is an NGO that wants to allocate funds effectively.  
We use clustering (unsupervised learning) to group countries based on socio-economic factors.
""")

# -----------------------------
# Dataset Description
# -----------------------------
st.markdown("""
# Dataset Description

- child_mort → Child mortality rate  
- exports → Export % of GDP  
- health → Health spending  
- imports → Import % of GDP  
- income → Net income  
- inflation → GDP growth rate  
- life_expec → Life expectancy  
- total_fer → Fertility rate  
- gdpp → GDP per capita  
""")

# -----------------------------
# Caching Functions (IMPORTANT)
# -----------------------------
@st.cache_data
def load_data():
    return conn.sql("""select * exclude country from country""").df()

@st.cache_data
def load_country_names():
    return conn.sql("""select country from country""").df()

@st.cache_data
def run_umap(data):
    model = UMAP(n_components=3, random_state=42)
    embed = model.fit_transform(data)
    return pd.DataFrame(embed)

@st.cache_data
def run_hdbscan(embed):
    model = HDBSCAN()
    model.fit(embed)
    return model.labels_

# -----------------------------
# Load Data
# -----------------------------
data = load_data()
countries = load_country_names()

# -----------------------------
# Data Exploration
# -----------------------------
st.header("Explore the Data 📊", divider='rainbow')

columns = conn.sql("from country limit 0").df().columns[1:]
stats = st.multiselect('Select features:', columns)

if stats:
    fig = make_subplots(rows=len(stats), cols=1,
                        subplot_titles=[s.upper() for s in stats])

    for i, col in enumerate(stats):
        temp = conn.sql(f"""
            select country, {col}
            from country
            order by {col} desc
        """).df()

        fig.add_trace(
            px.bar(temp, x='country', y=col).data[0],
            row=i+1, col=1
        )

    fig.update_layout(height=400*len(stats))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select features to explore")

# -----------------------------
# UMAP + HDBSCAN
# -----------------------------
embed = run_umap(data)
labels = run_hdbscan(embed)

# Combine
df = pd.concat([countries, embed], axis=1)
df['labels'] = labels

# Label Mapping
df['Class'] = 'Unknown'
df.loc[df['labels'] == -1, 'Class'] = 'Outlier'
df.loc[df['labels'] == 0, 'Class'] = 'Help Needed'
df.loc[df['labels'].isin([1, 2]), 'Class'] = 'May Need Help'
df.loc[df['labels'] >= 3, 'Class'] = 'Do Not Need Help'

# -----------------------------
# 3D Visualization
# -----------------------------
st.header("Clustering Visualization 🔍", divider='rainbow')

fig = px.scatter_3d(
    df,
    x=0,
    y=1,
    z=2,
    color='Class',
    hover_name='country'
)

fig.update_layout(height=700)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Map Visualization
# -----------------------------
st.header("Global Help Requirement 🌎")

fig_map = px.choropleth(
    df[['country', 'Class']],
    locations='country',
    locationmode='country names',
    color='Class',
    color_discrete_map={
        'Outlier': 'black',
        'Help Needed': 'red',
        'May Need Help': 'yellow',
        'Do Not Need Help': 'green'
    }
)

st.plotly_chart(fig_map, use_container_width=True)

# -----------------------------
# Conclusion
# -----------------------------
st.markdown("""
# Conclusion

Countries marked in **Red** require immediate help.  
Yellow indicates moderate need, Green indicates stable countries.
""")
