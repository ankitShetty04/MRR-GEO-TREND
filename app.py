import streamlit as st
import pandas as pd
import gspread
import json
from google.oauth2.service_account import Credentials

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
# Convert to DataFrame
df = pd.DataFrame(data)

# Convert 'Value' column to numeric
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
df = df.dropna(subset=['Value'])  # optional but recommended

# Convert Month and Churn Month columns to datetime
df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
df['Churn Month'] = pd.to_datetime(df['Churn Month'], errors='coerce')
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
                prev_mrr = group.iloc[i - 1]['Value']
                curr_mrr = group.iloc[i]['Value']
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
def get_churn_details(change_df, current_month):
    churned = change_df[change_df['Churn Month'].dt.date == current_month.date()]

    if churned.empty:
        return {'count': 0, 'mrr': 0, 'customers': '-'}

    id_to_name = churned.drop_duplicates(subset='NS ID').set_index('NS ID')['Name'].to_dict()

    return {
        'count': churned['NS ID'].nunique(),
        'mrr': churned['Curr_MRR'].sum(),
        'customers': ", ".join(sorted(id_to_name.values()))
    }

# Prepare summary
months = sorted(df_region['Month'].unique())

st.write("### MRR Expansion, Contraction & Churn Summary (±0.9% Threshold)")

table_data = []

for i, month in enumerate(months):
    if i == 0:
        table_data.append({
            'Month': month.strftime('%Y-%m'),
            'Churn Count': '',
            'Churn Customers': '',
            'Churn MRR': '',
            'Contraction Count': '',
            'Contraction Customers': '',
            'Contraction MRR': '',
            'Expansion Count': '',
            'Expansion Customers': '',
            'Expansion MRR': ''
        })
        continue

    month_changes = mrr_changes[mrr_changes['Month'] == month]

    expansion = get_change_details(month_changes, 'expansion')
    contraction = get_change_details(month_changes, 'contraction')
    churn = get_churn_details(month_changes, month)

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
        'Expansion MRR': f"₹{expansion['mrr']:,.2f}"
    })

summary_df = pd.DataFrame(table_data)

# Reorder columns
column_order = [
    'Month',
    'Churn Count', 'Churn Customers', 'Churn MRR',
    'Contraction Count', 'Contraction Customers', 'Contraction MRR',
    'Expansion Count', 'Expansion Customers', 'Expansion MRR'
]
summary_df = summary_df[column_order]

# Display table
st.dataframe(
    summary_df,
    use_container_width=True,
    column_config={
        'Month': st.column_config.TextColumn("Month", width="small"),
        'Churn Count': st.column_config.NumberColumn("Churn Count", width="small"),
        'Churn Customers': st.column_config.TextColumn("Churn Customers"),
        'Churn MRR': st.column_config.TextColumn("Churn MRR", width="small"),
        'Contraction Count': st.column_config.NumberColumn("Contraction Count", width="small"),
        'Contraction Customers': st.column_config.TextColumn("Contraction Customers"),
        'Contraction MRR': st.column_config.TextColumn("Contraction MRR", width="small"),
        'Expansion Count': st.column_config.NumberColumn("Expansion Count", width="small"),
        'Expansion Customers': st.column_config.TextColumn("Expansion Customers"),
        'Expansion MRR': st.column_config.TextColumn("Expansion MRR", width="small")
    }
)

# Footer
st.markdown("---")
st.caption("Note: Churn = Customers with Churn Month == Selected Month. Expansion = MRR ↑ > 0.9%, Contraction = MRR ↓ < -0.9%. First month excluded.")
