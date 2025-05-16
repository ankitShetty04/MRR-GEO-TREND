import streamlit as st
import pandas as pd
import gspread
import plotly.graph_objects as go
import json
from google.oauth2.service_account import Credentials

st.set_page_config(layout="wide")

# Define Google Sheet variables
SHEET_NAME = "Subscription"
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1Wgm4xVidpndlEKLXXZUylNGp0uJx_m11DBePOCMAz88/"

# Load credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
service_account_info = st.secrets["gcp_service_account"]  # ✅ Already a dict
creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
client = gspread.authorize(creds)

# Load sheet
sheet = client.open_by_url(SPREADSHEET_URL)
worksheet = sheet.worksheet(SHEET_NAME)
data = worksheet.get_all_records()

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert 'Value' column to numeric
df['Sub MRR'] = pd.to_numeric(df['Sub MRR'], errors='coerce')
df['1st Sub MRR'] = pd.to_numeric(df['1st Sub MRR'], errors='coerce')

df = df.dropna(subset=['Sub MRR'])  # optional but recommended

# Convert Month and Churn Month columns to datetime
df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
df['Churn Month'] = pd.to_datetime(df['Churn Month'], errors='coerce')
df['Churn Month MRR'] = pd.to_numeric(df['Churn Month MRR'], errors='coerce')


# Convert Month and 1st Sub Month columns to datetime
df['1st Sub Month'] = pd.to_datetime(df['1st Sub Month'],errors='coerce')

df = df.sort_values(['Name', 'Month'])

# Region filter
region_list = ['All'] + df['Region'].dropna().unique().tolist()
selected_region = st.selectbox("Select Region", region_list)

if selected_region == 'All':
    df_region = df.copy()
else:
    df_region = df[df['Region'] == selected_region]

# Function to calculate MRR changes
def get_mrr_changes(df):
    results = []
    for ns_id, group in df.groupby('NS ID'):
        group = group.sort_values('Month')
        name = group.iloc[0]['Name']
        if len(group) > 1:
            for i in range(1, len(group)):
                prev_mrr = group.iloc[i - 1]['Sub MRR']
                curr_mrr = group.iloc[i]['Sub MRR']
                if prev_mrr == 0:
                    pct_change = 0
                else:
                    pct_change = (curr_mrr - prev_mrr) / prev_mrr * 100

                results.append({
                    'NS ID': ns_id,
                    'Name': name,
                    'Month': group.iloc[i]['Month'],
                    'Prev_MRR': prev_mrr,
                    'Curr_MRR': curr_mrr,
                    'Pct_Change': pct_change,
                    'MRR_Change': curr_mrr - prev_mrr,
                    'Churn Month': group.iloc[i].get('Churn Month', pd.NaT)
                })

    return pd.DataFrame(results)

mrr_changes = get_mrr_changes(df_region)

# Expansion / Contraction
def get_change_details(change_df, change_type):
    valid_changes = change_df[change_df['Churn Month'].isna()]

    if change_type == 'expansion':
        filtered = valid_changes[valid_changes['Pct_Change'] > 0.9]
    else:
        filtered = valid_changes[valid_changes['Pct_Change'] < -0.9]

    if filtered.empty:
        return {'count': 0, 'customers': '-', 'mrr': 0}

    id_to_name = filtered.drop_duplicates(subset='NS ID').set_index('NS ID')['Name'].to_dict()

    return {
        'count': filtered['NS ID'].nunique(),
        'customers': ", ".join(sorted(id_to_name.values())),
        'mrr': filtered['MRR_Change'].sum()
    }

# Churn
def get_churn_details(df, current_month):
    churned = df[df['Churn Month'].notna() & (df['Churn Month'].dt.date == current_month.date())]

    if churned.empty:
        return {'count': 0, 'mrr': 0, 'customers': '-'}

    # Ensure 'Churn Month MRR' is numeric
    churned['Churn Month MRR'] = pd.to_numeric(churned['Churn Month MRR'], errors='coerce')

    id_to_name = churned.drop_duplicates(subset='NS ID').set_index('NS ID')['Name'].to_dict()
    mrr_sum = churned.groupby('NS ID')['Churn Month MRR'].first().sum()

    return {
        'count': churned['NS ID'].nunique(),
        'mrr': mrr_sum,
        'customers': ", ".join(sorted(id_to_name.values()))
    }


#1st Sub
def get_first_sub_details(df, month):
    first_subs = df[df['1st Sub Month'].notna() & (df['1st Sub Month'].dt.date == month.date())]

    if first_subs.empty:
        return {'count': 0, 'mrr': 0, 'customers': '-'}

    id_to_name = first_subs.drop_duplicates(subset='NS ID').set_index('NS ID')['Name'].to_dict()

    # Use '1st Sub MRR' instead of 'Value'
    mrr_sum = first_subs.groupby('NS ID')['1st Sub MRR'].first().sum()

    return {
        'count': first_subs['NS ID'].nunique(),
        'mrr': mrr_sum,
        'customers': ", ".join(sorted(id_to_name.values()))
    }


def get_closing_metrics(df, month):
    # Customers who are active in the current month:
    # - Not churned yet or churned after this month
    condition_churn = (df['Churn Month'].isna()) | (df['Churn Month'] > month)

    # - Subscribed on or before this month
    condition_sub = (df['1st Sub Month'].isna()) | (df['1st Sub Month'] <= month)

    # Filter to active customers for this specific month
    closing_df = df[condition_churn & condition_sub]
    month_df = closing_df[closing_df['Month'] == month]

    closing_cx_count = month_df['NS ID'].nunique()
    closing_cx_mrr_count = month_df[month_df['Sub MRR'] > 0]['NS ID'].nunique()
    closing_mrr = month_df.groupby('NS ID')['Sub MRR'].first().sum()

    return {
        'closing_cx_count': closing_cx_count,
        'closing_cx_mrr_count': closing_cx_mrr_count,
        'closing_mrr': closing_mrr
    }


# Prepare summary
months = sorted(df_region['Month'].unique())

st.write("### MRR GEO Trend (±0.9% Threshold)")
st.caption("Note: Churn = Customers with Churn Month == Selected Month. Expansion = MRR ↑ > 0.9%, Contraction = MRR ↓ < -0.9%. First month excluded.")

table_data = []

for i, month in enumerate(months):
    if i == 0:
    month_changes = mrr_changes[mrr_changes['Month'] == month]
    churn = get_churn_details(df_region, month)
    first_sub = get_first_sub_details(df_region, month)
    closing = get_closing_metrics(df_region, month)

    table_data.append({
        'Month': month.strftime('%Y-%m'),
        'Churn Count': churn['count'],
        'Churn Customers': churn['customers'],
        'Churn MRR': f"₹{churn['mrr']:,.2f}",
        'Contraction Count': '',
        'Contraction Customers': '',
        'Contraction MRR': '',
        'Expansion Count': '',
        'Expansion Customers': '',
        'Expansion MRR': '',
        '1st Sub Count': first_sub['count'],
        '1st Sub Customers': first_sub['customers'],
        '1st Sub MRR': f"₹{first_sub['mrr']:,.2f}",
        'Closing Cx Count': closing['closing_cx_count'],
        'Closing Cx MRR Count': closing['closing_cx_mrr_count'],
        'Closing MRR': f"₹{closing['closing_mrr']:,.2f}"
    })
    continue

    month_changes = mrr_changes[mrr_changes['Month'] == month]

    expansion = get_change_details(month_changes, 'expansion')
    contraction = get_change_details(month_changes, 'contraction')
    first_sub = get_first_sub_details(df_region, month)
    churn = get_churn_details(df_region, month)
    closing = get_closing_metrics(df_region, month)

    table_data.append({
        'Month': month.strftime('%Y-%m'),
        'Churn Count': churn['count'],
        'Churn Customers': churn['customers'],
        'Churn MRR': f"₹{churn['mrr']:,.2f}",
        'Contraction Count': contraction['count'],
        'Contraction Customers': contraction['customers'],
        'Contraction MRR': f"₹{contraction['mrr']:,.2f}",
        'Expansion Count': expansion['count'],
        'Expansion Customers': expansion['customers'],
        'Expansion MRR': f"₹{expansion['mrr']:,.2f}",
        '1st Sub Count': first_sub['count'],
        '1st Sub Customers': first_sub['customers'],
        '1st Sub MRR': f"₹{first_sub['mrr']:,.2f}",
        'Closing Cx Count': closing['closing_cx_count'],
        'Closing Cx MRR Count': closing['closing_cx_mrr_count'],
        'Closing MRR': f"₹{closing['closing_mrr']:,.2f}"
    })


summary_df = pd.DataFrame(table_data)

# Reorder columns
column_order = [
    'Month',
    'Churn Count', 'Churn Customers', 'Churn MRR',
    'Contraction Count', 'Contraction Customers', 'Contraction MRR',
    'Expansion Count', 'Expansion Customers', 'Expansion MRR',
    '1st Sub Count', '1st Sub Customers', '1st Sub MRR',
    'Closing Cx Count', 'Closing Cx MRR Count', 'Closing MRR'
]

summary_df = summary_df[column_order]

# Display table
st.dataframe(
    summary_df,
    use_container_width=True,
    column_config={
        'Month': st.column_config.TextColumn("Month", width="small"),

        'Churn Count': st.column_config.NumberColumn("Churn Count", width="tiny"),
        'Churn Customers': st.column_config.TextColumn("Churn Customers"),  # keep default for wrapping
        'Churn MRR': st.column_config.TextColumn("Churn MRR", width="tiny"),

        'Contraction Count': st.column_config.NumberColumn("Contraction Count", width="tiny"),
        'Contraction Customers': st.column_config.TextColumn("Contraction Customers"),
        'Contraction MRR': st.column_config.TextColumn("Contraction MRR", width="tiny"),

        'Expansion Count': st.column_config.NumberColumn("Expansion Count", width="tiny"),
        'Expansion Customers': st.column_config.TextColumn("Expansion Customers"),
        'Expansion MRR': st.column_config.TextColumn("Expansion MRR", width="tiny"),

        '1st Sub Count': st.column_config.NumberColumn("1st Sub Count", width="tiny"),
        '1st Sub Customers': st.column_config.TextColumn("1st Sub Customers"),
        '1st Sub MRR': st.column_config.TextColumn("1st Sub MRR", width="tiny"),

        'Closing Cx Count': st.column_config.NumberColumn("Closing Cx Count", width="tiny"),
        'Closing Cx MRR Count': st.column_config.NumberColumn("Closing Cx MRR Count", width="tiny"),
        'Closing MRR': st.column_config.TextColumn("Closing MRR", width="tiny"),

    }
)


st.markdown("---")

st.markdown("### Sub x Cons Trend by Region")
# Ensure Cons MRR is numeric
df_region['Cons MRR'] = pd.to_numeric(df_region.get('Cons MRR'), errors='coerce')
# Group by Month and aggregate Sub MRR & Cons MRR
trend_df = df_region.groupby('Month', as_index=False).agg({
    'Sub MRR': 'sum',
    'Cons MRR': 'sum'
}).sort_values('Month')


# Create plotly figure
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=trend_df['Month'],
    y=trend_df['Sub MRR'],
    mode='lines+markers+text',
    name='Sub MRR',
    line=dict(color='green', width=2),
    marker=dict(size=6),
    text=(trend_df['Sub MRR'] / 1e7).round(1),  # Value in crores
    textposition='top center',
    texttemplate='%{text}',
    hovertemplate='Month: %{x|%b-%Y}<br>Sub MRR: ₹%{y:,.0f}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=trend_df['Month'],
    y=trend_df['Cons MRR'],
    mode='lines+markers+text',
    name='Cons MRR',
    line=dict(color='darkorange', width=2, dash='dot'),
    marker=dict(size=6),
    text=(trend_df['Cons MRR'] / 1e7).round(1),  # Value in crores
    textposition='top center',
    texttemplate='%{text}',
    hovertemplate='Month: %{x|%b-%Y}<br>Cons MRR: ₹%{y:,.0f}<extra></extra>'
))


fig.update_layout(
    title=f"SUB x Cons Trend by Month ({selected_region} Region)",
    xaxis_title="Month",
    yaxis_title="MRR (₹)",
    xaxis=dict(tickformat="%b-%Y"),
    hovermode='x unified',
    template='plotly_white',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    margin=dict(l=20, r=20, t=60, b=20)
)

# Show chart
st.plotly_chart(fig, use_container_width=True)


# ---- Director AM based Chart ----
st.markdown("### Sub x Cons Trend by Director AM (Independent of Region Filter)")

# Get all unique Director AMs
all_directors = df['Director AM'].dropna().unique().tolist()
selected_director = st.selectbox("Select Director AM", all_directors, key="director_am_selector")

# Filter original df (not region-filtered one)
df_director = df[df['Director AM'] == selected_director]

# Group by Month and aggregate Sub MRR & Cons MRR
trend_director_df = df_director.groupby('Month', as_index=False).agg({
    'Sub MRR': 'sum',
    'Cons MRR': 'sum'
}).sort_values('Month')

# Create a second chart for Director AM
fig_director = go.Figure()

fig_director.add_trace(go.Scatter(
    x=trend_df['Month'],
    y=trend_df['Sub MRR'],
    mode='lines+markers+text',
    name='Sub MRR',
    line=dict(color='green', width=2),
    marker=dict(size=6),
    text=(trend_df['Sub MRR'] / 1e7).round(1),  # Value in crores
    textposition='top center',
    texttemplate='%{text}',
    hovertemplate='Month: %{x|%b-%Y}<br>Sub MRR: ₹%{y:,.0f}<extra></extra>'
))


fig_director.add_trace(go.Scatter(
    x=trend_df['Month'],
    y=trend_df['Cons MRR'],
    mode='lines+markers+text',
    name='Cons MRR',
    line=dict(color='darkorange', width=2, dash='dot'),
    marker=dict(size=6),
    text=(trend_df['Cons MRR'] / 1e7).round(1),  # Value in crores
    textposition='top center',
    texttemplate='%{text}',
    hovertemplate='Month: %{x|%b-%Y}<br>Cons MRR: ₹%{y:,.0f}<extra></extra>'
))


fig_director.update_layout(
    title=f"MRR Trend by Month for Director AM: {selected_director}",
    xaxis_title="Month",
    yaxis_title="MRR (₹)",
    xaxis=dict(tickformat="%b-%Y"),
    hovermode='x unified',
    template='plotly_white',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(fig_director, use_container_width=True)


# Footer
