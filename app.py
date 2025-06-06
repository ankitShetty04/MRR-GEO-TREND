import streamlit as st
import pandas as pd
import gspread
import plotly.graph_objects as go
import json
import seaborn as sns
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


# Password protected directors
PROTECTED_DIRECTORS = {
    "Subodh Selvaraj": "subodh123",
    "Aditya Joshi": "aditya123",
    "Shreya Trivedi": "shreya123",
    "Aditya Mahajan": "mahajan123",
    "Yash Kapoor": "yash123"
}

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["MRR Geo Trend", "Director AM Analysis"])


def get_closing_metrics(df, month):
    """Calculate closing customer metrics for a given month"""
    # Ensure month is Timestamp if it's not already
    if not isinstance(month, pd.Timestamp):
        month = pd.to_datetime(month)

    # Convert 'Churn Month' and '1st Sub Month' to datetime if they're not already
    df = df.copy()  # Avoid modifying original dataframe
    df['Churn Month'] = pd.to_datetime(df['Churn Month'], errors='coerce')
    df['1st Sub Month'] = pd.to_datetime(df['1st Sub Month'], errors='coerce')

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

 # 1st Sub
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



if page == "MRR Geo Trend":
    # Your existing code for the MRR Geo Trend page
    # Convert 'Sub MRR' column to numeric
    df['Sub MRR'] = pd.to_numeric(df['Sub MRR'], errors='coerce')
    df['1st Sub MRR'] = pd.to_numeric(df['1st Sub MRR'], errors='coerce')
    df['Cons MRR'] = pd.to_numeric(df['Cons MRR'], errors='coerce')

    df = df.dropna(subset=['Sub MRR'])  # optional but recommended

    # Convert Month and Churn Month columns to datetime
    df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
    df['Churn Month'] = pd.to_datetime(df['Churn Month'], errors='coerce')
    df['Churn Month MRR'] = pd.to_numeric(df['Churn Month MRR'], errors='coerce')

    # Convert Month and 1st Sub Month columns to datetime
    df['1st Sub Month'] = pd.to_datetime(df['1st Sub Month'], errors='coerce')

    df = df.sort_values(['Name', 'Month'])

    # Region filter
    region_list = ['All'] + df['Region'].dropna().unique().tolist()
    selected_region = st.selectbox("Select Region", region_list)

    # Region-wise Category filter - dynamic based on Region selection
    if selected_region == 'All':
        category_list = ['All'] + df['Region-wise Category'].dropna().unique().tolist()
    else:
        available_categories = df[df['Region'] == selected_region]['Region-wise Category'].dropna().unique().tolist()
        category_list = ['All'] + available_categories

    selected_category = st.selectbox("Select Region-wise Category", category_list)

    # Apply both filters to the data
    if selected_region == 'All' and selected_category == 'All':
        df_filtered = df.copy()
    elif selected_region == 'All':
        df_filtered = df[df['Region-wise Category'] == selected_category]
    elif selected_category == 'All':
        df_filtered = df[df['Region'] == selected_region]
    else:
        df_filtered = df[(df['Region'] == selected_region) &
                         (df['Region-wise Category'] == selected_category)]


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


    mrr_changes = get_mrr_changes(df_filtered)


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


    # 1st Sub
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




    def get_consumables_cx_count(df, month):
        """Count customers with Cons MRR > 0 for the given month"""
        active_in_month = df[df['Month'] == month]
        consumables_cx = active_in_month[active_in_month['Cons MRR'] > 0]
        return consumables_cx['NS ID'].nunique()


    def get_consumables_mrr(df, month):
        """Sum of all Cons MRR for the given month"""
        active_in_month = df[df['Month'] == month]
        return active_in_month['Cons MRR'].sum()


    # Prepare summary
    months = sorted(df_filtered['Month'].unique())

    st.write("### MRR GEO Trend (±0.9% Threshold)")
    st.caption(
        "Note: Churn = Customers with Churn Month == Selected Month. Expansion = MRR ↑ > 0.9%, Contraction = MRR ↓ < -0.9%. First month excluded.")

    table_data = []

    for i, month in enumerate(months):
        if i == 0:
            month_changes = mrr_changes[mrr_changes['Month'] == month]
            churn = get_churn_details(df_filtered, month)
            first_sub = get_first_sub_details(df_filtered, month)
            Close_sec = get_closing_metrics(df_filtered, month)
            Cons_mrr_Cx = get_consumables_cx_count(df_filtered, month)
            Cons_mrr = get_consumables_mrr(df_filtered, month)
            # Opening Cx count
            closing = get_closing_metrics(df_filtered, month)
            opening_cx_count = closing['closing_cx_count'] + churn['count'] - first_sub['count']
            opening_cx_mrr_count = closing['closing_cx_mrr_count'] + churn['count'] - first_sub['count']
            opening_cx_mrr = closing['closing_mrr'] + churn['mrr'] - first_sub['mrr']

            table_data.append({
                'Month': month.strftime('%Y-%m'),
                'Opening Cx Count': opening_cx_count,
                'Opening Cx MRR Count': opening_cx_mrr_count,
                'Opening Cx MRR': f"₹{opening_cx_mrr:,.2f}",
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
                'Closing Cx Count': Close_sec['closing_cx_count'],
                'Closing Cx MRR Count': Close_sec['closing_cx_mrr_count'],
                'Closing MRR': f"₹{Close_sec['closing_mrr']:,.2f}",
                'Consumables Cx Count': Cons_mrr_Cx,
                'Consumables MRR': f"₹{Cons_mrr:,.2f}",
                'Total MRR': f"₹{Close_sec['closing_mrr'] + Cons_mrr:,.2f}"
            })
            continue

        month_changes = mrr_changes[mrr_changes['Month'] == month]

        expansion = get_change_details(month_changes, 'expansion')
        contraction = get_change_details(month_changes, 'contraction')
        first_sub = get_first_sub_details(df_filtered, month)
        churn = get_churn_details(df_filtered, month)
        closing = get_closing_metrics(df_filtered, month)
        # Get consumables metrics using the new functions
        consumables_cx_count = get_consumables_cx_count(df_filtered, month)
        consumables_mrr = get_consumables_mrr(df_filtered, month)
        # Get Cx Opening
        opening_cx_count = closing['closing_cx_count'] + churn['count'] - first_sub['count']
        opening_cx_mrr_count = closing['closing_cx_mrr_count'] + churn['count'] - first_sub['count']
        opening_cx_mrr = closing['closing_mrr'] + churn['mrr'] - first_sub['mrr']

        table_data.append({
            'Month': month.strftime('%Y-%m'),
            'Opening Cx Count': opening_cx_count,
            'Opening Cx MRR Count': opening_cx_mrr_count,
            'Opening Cx MRR': f"₹{opening_cx_mrr:,.2f}",
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
            'Closing MRR': f"₹{closing['closing_mrr']:,.2f}",
            'Consumables Cx Count': consumables_cx_count,
            'Consumables MRR': f"₹{consumables_mrr:,.2f}",
            'Total MRR': f"₹{closing['closing_mrr'] + consumables_mrr:,.2f}"
        })

    summary_df = pd.DataFrame(table_data)

    # Reorder columns
    column_order = [
        'Month', 'Opening Cx Count', 'Opening Cx MRR Count', 'Opening Cx MRR',
        'Churn Count', 'Churn Customers', 'Churn MRR',
        'Contraction Count', 'Contraction Customers', 'Contraction MRR',
        'Expansion Count', 'Expansion Customers', 'Expansion MRR',
        '1st Sub Count', '1st Sub Customers', '1st Sub MRR',
        'Closing Cx Count', 'Closing Cx MRR Count', 'Closing MRR', 'Consumables Cx Count', 'Consumables MRR',
        'Total MRR'
    ]

    summary_df = summary_df[column_order]

    # Display table
    st.dataframe(
        summary_df,
        use_container_width=True,
        column_config={
            'Month': st.column_config.TextColumn("Month", width="small"),

            'Opening Cx Count': st.column_config.NumberColumn("Opening Cx Count", width="tiny"),
            'Opening Cx MRR Count': st.column_config.NumberColumn("Opening Cx with MRR", width="tiny"),
            'Opening Cx MRR': st.column_config.TextColumn("Opening Cx MRR", width="tiny"),

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

            'Consumables Cx Count': st.column_config.NumberColumn("Consumables Cx", width="tiny"),
            'Consumables MRR': st.column_config.TextColumn("Consumables MRR", width="tiny")
        }
    )

    st.markdown("---")



    # Ensure Cons MRR is numeric
    df_filtered['Cons MRR'] = pd.to_numeric(df_filtered.get('Cons MRR'), errors='coerce')
    # Group by Month and aggregate Sub MRR & Cons MRR
    trend_df = df_filtered.groupby('Month', as_index=False).agg({
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
        title=f"SUB x Cons Trend by Month (Region: {selected_region}, Category: {selected_category})",
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

elif page == "Director AM Analysis":
    st.title("Director AM Analysis")

    # Get all unique Director AMs
    all_directors = df['Director AM'].dropna().unique().tolist()
    selected_director = st.selectbox("Select Director AM", all_directors, key="director_am_selector")

    # Check if selected director is protected
    if selected_director in PROTECTED_DIRECTORS:
        password = st.text_input(f"Enter password for {selected_director}", type="password")

        if password != PROTECTED_DIRECTORS[selected_director]:
            st.warning("Please enter the correct password to view this director's data")
            st.stop()

    # Filter original df for selected director
    df_director = df[df['Director AM'] == selected_director]
    df_director['Month'] = pd.to_datetime(df_director['Month'], errors='coerce')
    df_director['Churn Month'] = pd.to_datetime(df_director['Churn Month'], errors='coerce')
    df_director['1st Sub Month'] = pd.to_datetime(df_director['1st Sub Month'], errors='coerce')
    df_director['Sub MRR'] = pd.to_numeric(df_director['Sub MRR'], errors='coerce')



    df_director['Sub MRR'] = pd.to_numeric(df_director.get('Sub MRR'), errors='coerce')
    df_director['Cons MRR'] = pd.to_numeric(df_director.get('Cons MRR'), errors='coerce')

    # Convert Month to datetime if not already
    df_director['Month'] = pd.to_datetime(df_director['Month'])
    months = sorted(df_director['Month'].unique())

    # ---- Director AM based Chart ----
    st.markdown("### Sub x Cons Trend by Director AM")

    # Group by Month and aggregate Sub MRR & Cons MRR
    trend_director_df = df_director.groupby('Month', as_index=False).agg({
        'Sub MRR': 'sum',
        'Cons MRR': 'sum'
    }).sort_values('Month')

    # Create a chart for Director AM
    fig_director = go.Figure()

    fig_director.add_trace(go.Scatter(
        x=trend_director_df['Month'],
        y=trend_director_df['Sub MRR'],
        mode='lines+markers+text',
        name='Sub MRR',
        line=dict(color='green', width=2),
        marker=dict(size=6),
        text=(trend_director_df['Sub MRR'] / 1e7).round(1),  # Value in crores
        textposition='top center',
        texttemplate='%{text}',
        hovertemplate='Month: %{x|%b-%Y}<br>Sub MRR: ₹%{y:,.0f}<extra></extra>'
    ))

    fig_director.add_trace(go.Scatter(
        x=trend_director_df['Month'],
        y=trend_director_df['Cons MRR'],
        mode='lines+markers+text',
        name='Cons MRR',
        line=dict(color='darkorange', width=2, dash='dot'),
        marker=dict(size=6),
        text=(trend_director_df['Cons MRR'] / 1e7).round(1),  # Value in crores
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

    # ---- Director AM based Customer Count Trend ----
    st.markdown("### Opening vs Closing Customer Count by Director AM")

    # Prepare customer count data from the table (but filtered by Director AM)
    director_summary_data = []

    for i, month in enumerate(months):
        if i == 0:
            # For first month, we can't calculate changes

            closing = get_closing_metrics(df_director, month)
            churn = get_churn_details(df_director, month)
            first_sub = get_first_sub_details(df_director, month)

            opening_cx_count = closing['closing_cx_count'] + churn['count'] - first_sub['count']
            closing_cx_count = closing['closing_cx_count']

            director_summary_data.append({
                'Month': month,
                'Opening Cx Count': opening_cx_count,
                'Closing Cx Count': closing_cx_count
            })
            continue

        # For subsequent months
        closing = get_closing_metrics(df_director, month)
        churn = get_churn_details(df_director, month)
        first_sub = get_first_sub_details(df_director, month)

        opening_cx_count = closing['closing_cx_count'] + churn['count'] - first_sub['count']
        closing_cx_count = closing['closing_cx_count']

        director_summary_data.append({
            'Month': month,
            'Opening Cx Count': opening_cx_count,
            'Closing Cx Count': closing_cx_count
        })

    # Create DataFrame
    director_cx_trend = pd.DataFrame(director_summary_data)

    # Create plotly figure
    fig_cx_trend = go.Figure()

    fig_cx_trend.add_trace(go.Scatter(
        x=director_cx_trend['Month'],
        y=director_cx_trend['Opening Cx Count'],
        mode='lines+markers+text',
        name='Opening Customers',
        line=dict(color='blue', width=2),
        marker=dict(size=8),
        text=director_cx_trend['Opening Cx Count'],
        textposition='top center',
        hovertemplate='Month: %{x|%b-%Y}<br>Opening Customers: %{y}<extra></extra>'
    ))

    fig_cx_trend.add_trace(go.Scatter(
        x=director_cx_trend['Month'],
        y=director_cx_trend['Closing Cx Count'],
        mode='lines+markers+text',
        name='Closing Customers',
        line=dict(color='green', width=2, dash='dot'),
        marker=dict(size=8),
        text=director_cx_trend['Closing Cx Count'],
        textposition='bottom center',
        hovertemplate='Month: %{x|%b-%Y}<br>Closing Customers: %{y}<extra></extra>'
    ))

    fig_cx_trend.update_layout(
        title=f"Customer Count Trend for Director AM: {selected_director}",
        xaxis_title="Month",
        yaxis_title="Customer Count",
        xaxis=dict(tickformat="%b-%Y"),
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig_cx_trend, use_container_width=True)

    # ---- Director AM based MRR Trend ----
    st.markdown("### Opening vs Closing MRR by Director AM")

    # Prepare MRR data from the table (filtered by Director AM)
    director_mrr_data = []

    for i, month in enumerate(months):
        if i == 0:
            # For first month
            closing = get_closing_metrics(df_director, month)
            churn = get_churn_details(df_director, month)
            first_sub = get_first_sub_details(df_director, month)

            opening_mrr = closing['closing_mrr'] + churn['mrr'] - first_sub['mrr']
            closing_mrr = closing['closing_mrr']

            director_mrr_data.append({
                'Month': month,
                'Opening MRR': opening_mrr,
                'Closing MRR': closing_mrr
            })
            continue

        # For subsequent months
        closing = get_closing_metrics(df_director, month)
        churn = get_churn_details(df_director, month)
        first_sub = get_first_sub_details(df_director, month)

        opening_mrr = closing['closing_mrr'] + churn['mrr'] - first_sub['mrr']
        closing_mrr = closing['closing_mrr']

        director_mrr_data.append({
            'Month': month,
            'Opening MRR': opening_mrr,
            'Closing MRR': closing_mrr
        })

    # Create DataFrame
    director_mrr_trend = pd.DataFrame(director_mrr_data)

    # Create plotly figure
    fig_mrr_trend = go.Figure()

    fig_mrr_trend.add_trace(go.Scatter(
        x=director_mrr_trend['Month'],
        y=director_mrr_trend['Opening MRR'],
        mode='lines+markers+text',
        name='Opening MRR',
        line=dict(color='blue', width=2),
        marker=dict(size=8),
        text=(director_mrr_trend['Opening MRR'] / 1e5).round(1).astype(str) + 'L',
        textposition='top center',
        hovertemplate='Month: %{x|%b-%Y}<br>Opening MRR: ₹%{y:,.0f}<extra></extra>'
    ))

    fig_mrr_trend.add_trace(go.Scatter(
        x=director_mrr_trend['Month'],
        y=director_mrr_trend['Closing MRR'],
        mode='lines+markers+text',
        name='Closing MRR',
        line=dict(color='green', width=2, dash='dot'),
        marker=dict(size=8),
        text=(director_mrr_trend['Closing MRR'] / 1e5).round(1).astype(str) + 'L',
        textposition='bottom center',
        hovertemplate='Month: %{x|%b-%Y}<br>Closing MRR: ₹%{y:,.0f}<extra></extra>'
    ))

    fig_mrr_trend.update_layout(
        title=f"MRR Trend for Director AM: {selected_director}",
        xaxis_title="Month",
        yaxis_title="MRR (₹)",
        xaxis=dict(tickformat="%b-%Y"),
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig_mrr_trend, use_container_width=True)

st.markdown("""
    <style>
    .stApp {
        background-color: #f5f7fa;
        color: #2c3e50;
    }

    h1, h2, h3 {
        color: #1f3a93;
    }

    div[data-testid="metric-container"] {
        background-color: #ffffff;
        color: #2c3e50;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border: 1px solid #dcdde1;
    }

    .stButton>button {
        background-color: #34495e;
        color: white;
        border-radius: 6px;
    }

    .stButton>button:hover {
        background-color: #5d6d7e;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)
