import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# डेटा क्लीनिंग फंक्शन
def clean_and_prepare_data(df, shift_columns):
    all_data = []
    # सुनिश्चित करें कि डेट कॉलम सही से लोड हो रहा है
    # हम B कॉलम (Index 1) को डेट मान रहे हैं
    for index, row in df.iterrows():
        try:
            # पहली कुछ लाइन्स छोड़ने के लिए अगर डेटा नहीं है
            if pd.isna(row.iloc[1]): continue 
            
            current_date = pd.to_datetime(row.iloc[1])
            day_of_week = current_date.dayofweek
            
            for shift_name in shift_columns:
                val = str(row[shift_name]).strip()
                if val.isdigit():
                    num = int(val)
                    if 0 <= num <= 99:
                        all_data.append({
                            'date': current_date,
                            'day_of_week': day_of_week,
                            'shift': shift_name,
                            'number': num
                        })
        except:
            continue
    return pd.DataFrame(all_data)

st.title("🎯 AI Number Predictor (00-99)")

uploaded_file = st.file_uploader("अपनी Excel फाइल अपलोड करें", type=['xlsx'])

if uploaded_file:
    try:
        # एक्सेल लोड करना
        df = pd.read_excel(uploaded_file)
        
        # शिफ्ट कॉलम्स (C1 से I1 यानी Index 2 से 8)
        shift_cols = df.columns[2:9]
        
        target_shift = st.selectbox("शिफ्ट चुनें:", shift_cols)

        if st.button("🔮 Predict"):
            clean_df = clean_and_prepare_data(df, shift_cols)
            
            if clean_df.empty:
                st.error("डेटा नहीं मिला! कृपया चेक करें कि एक्सेल में नंबर्स 00-99 के बीच हैं।")
            else:
                # प्रेडिक्शन लॉजिक (Random Forest)
                # ... (बाकी पुराना ट्रेनिंग कोड यहाँ रहेगा)
                st.success("प्रेडिक्शन तैयार है!")
    except Exception as e:
        st.error(f"Error: {e}")
        
