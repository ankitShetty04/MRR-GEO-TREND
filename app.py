import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# Define Google Sheet variables
SHEET_NAME = "Subscription"
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1Wgm4xVidpndlEKLXXZUylNGp0uJx_m11DBePOCMAz88/"

# Load credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_file("/Users/webengage/PycharmProjects/MRR Geo Tren/venv/utils/cred.json", scopes=scope)
client = gspread.authorize(creds)

# Load sheet
sheet = client.open_by_url(SPREADSHEET_URL)
worksheet = sheet.worksheet(SHEET_NAME)
data = worksheet.get_all_records()

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert 'Value' column to numeric
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

# Drop rows where Value is NaN (optional but recommended)
df = df.dropna(subset=['Value'])

# Ensure 'Month' is in datetime format and sort
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values(['Name', 'Month'])

# Ensure 'Month' is in datetime format and sort
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values(['Name', 'Month'])

# Filter selector - add "All" option
region_list = ['All'] + df['Region'].dropna().unique().tolist()
selected_region = st.selectbox("Select Region", region_list)

# Filter dataframe based on selection
if selected_region == 'All':
    df_region = df.copy()
else:
    df_region = df[df['Region'] == selected_region]

# Function to calculate % change and classify expansion/contraction
def get_mrr_changes(df):
    results = []
    for name, group in df.groupby('Name'):
        group = group.sort_values('Month')
        if len(group) > 1:
            for i in range(1, len(group)):
                prev_mrr = group.iloc[i - 1]['Value']
                curr_mrr = group.iloc[i]['Value']
                if prev_mrr == 0:
                    pct_change = 0
                else:
                    pct_change = (curr_mrr - prev_mrr) / prev_mrr * 100

                results.append({
                    'Name': name,
                    'Month': group.iloc[i]['Month'],
                    'Prev_MRR': prev_mrr,
                    'Curr_MRR': curr_mrr,
                    'Pct_Change': pct_change,
                    'MRR_Change': curr_mrr - prev_mrr,
                    'Churn Month': group.iloc[i].get('Churn Month', '-')
                })

    return pd.DataFrame(results)

# Calculate MRR changes for the selected region
mrr_changes = get_mrr_changes(df_region)

# Function to get change details with Churn Month condition
def get_change_details(change_df, change_type):
    # Only consider rows where Churn Month == '-'
    valid_changes = change_df[change_df['Churn Month'] == '-']

    if change_type == 'expansion':
        filtered = valid_changes[valid_changes['Pct_Change'] > 0.9]
    else:
        filtered = valid_changes[valid_changes['Pct_Change'] < -0.9]

    if filtered.empty:
        return {'count': 0, 'customers': '-', 'mrr': 0}

    return {
        'count': filtered['Name'].nunique(),
        'customers': ", ".join(sorted(filtered['Name'].unique())),
        'mrr': filtered['MRR_Change'].sum()
    }

# Get unique months
months = sorted(df_region['Month'].unique())

# Output header
st.write("### MRR Expansion & Contraction Summary (±0.9% Threshold)")

# Create a list to hold all rows for the table
table_data = []

for i, month in enumerate(months):
    if i == 0:
        # Skip first month — no prior data available
        table_data.append({
            'Month': month.strftime('%Y-%m'),
            'Expansion Count': '',
            'Expansion Customers': '',
            'Expansion MRR': '',
            'Contraction Count': '',
            'Contraction Customers': '',
            'Contraction MRR': ''
        })
        continue

    month_changes = mrr_changes[mrr_changes['Month'] == month]

    # Get expansion and contraction details
    expansion = get_change_details(month_changes, 'expansion')
    contraction = get_change_details(month_changes, 'contraction')

    table_data.append({
        'Month': month.strftime('%Y-%m'),
        'Expansion Count': expansion['count'],
        'Expansion Customers': expansion['customers'],
        'Expansion MRR': f"₹{expansion['mrr']:,.2f}",
        'Contraction Count': contraction['count'],
        'Contraction Customers': contraction['customers'],
        'Contraction MRR': f"₹{contraction['mrr']:,.2f}"
    })

# Convert to DataFrame
summary_df = pd.DataFrame(table_data)

# Reorder columns for better presentation
column_order = [
    'Month',
    'Contraction Count', 'Contraction Customers', 'Contraction MRR',
    'Expansion Count', 'Expansion Customers', 'Expansion MRR'
]
summary_df = summary_df[column_order]

# Display as table with improved formatting
st.dataframe(
    summary_df,
    use_container_width=True,
    column_config={
        'Month': st.column_config.TextColumn("Month", width="small"),
        'Expansion Count': st.column_config.NumberColumn("Expansion Count", width="small"),
        'Expansion Customers': st.column_config.TextColumn("Expansion Customers"),
        'Expansion MRR': st.column_config.TextColumn("Expansion MRR", width="small"),
        'Contraction Count': st.column_config.NumberColumn("Contraction Count", width="small"),
        'Contraction Customers': st.column_config.TextColumn("Contraction Customers"),
        'Contraction MRR': st.column_config.TextColumn("Contraction MRR", width="small")
    }
)

# Add some visual separation
st.markdown("---")
st.caption("Note: Expansion = MRR increase > 0.9%, Contraction = MRR decrease < -0.9%. First month is excluded.")
