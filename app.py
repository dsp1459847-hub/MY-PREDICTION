import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# डेटा क्लीनिंग फंक्शन: सिर्फ 00-99 के नंबर्स को प्रोसेस करेगा
def clean_and_prepare_data(df, shift_columns):
    all_data = []
    
    for index, row in df.iterrows():
        # मान लें कि B1 कॉलम का नाम 'date' है
        try:
            current_date = pd.to_datetime(row.iloc[1]) # B1 कॉलम (Index 1)
            day_of_week = current_date.dayofweek
            
            for shift_name in shift_columns:
                val = str(row[shift_name]).strip()
                
                # टेक्स्ट (XX), खाली सेल या अन्य शब्दों को हटाना
                if val.isdigit():
                    num = int(val)
                    if 0 <= num <= 99: # 00 से 99 की रेंज
                        all_data.append({
                            'date': current_date,
                            'day_of_week': day_of_week,
                            'shift': shift_name,
                            'number': num
                        })
        except:
            continue
            
    clean_df = pd.DataFrame(all_data)
    if not clean_df.empty:
        clean_df = clean_df.sort_values(['date', 'shift'])
    return clean_df

# ट्रेनिंग और टॉप 3 प्रेडिक्शन
def train_and_predict(clean_df, target_shift):
    shift_list = list(clean_df['shift'].unique())
    clean_df['shift_id'] = clean_df['shift'].apply(lambda x: shift_list.index(x))
    
    window = 5 
    X, y = [], []
    nums = clean_df['number'].values
    days = clean_df['day_of_week'].values
    shifts = clean_df['shift_id'].values
    
    for i in range(window, len(clean_df)):
        features = list(nums[i-window:i]) + [days[i], shifts[i]]
        X.append(features)
        y.append(nums[i])
        
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X, y)
    
    target_shift_id = shift_list.index(target_shift)
    # आज का दिन या कल का दिन लॉजिक
    next_day = days[-1] 
    last_features = list(nums[-window:]) + [next_day, target_shift_id]
    
    probs = model.predict_proba([last_features])[0]
    top_3_indices = np.argsort(probs)[-3:][::-1]
    
    predictions = []
    for idx in top_3_indices:
        predictions.append({
            'number': model.classes_[idx],
            'prob': probs[idx] * 100
        })
    return predictions

# --- UI Setup ---
st.set_page_config(page_title="00-99 Number Predictor", layout="wide")
st.title("🎯 AI Number Guessing (00-99 Range)")
st.write("A1: Serial, B1: Date, C1-I1: Shifts (09:00 AM to 08:00 AM)")

uploaded_file = st.file_uploader("अपनी Excel फाइल यहाँ डालें", type=['xlsx'])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    # कॉलम्स को C1 से I1 (Index 2 से 8) के रूप में लेना
    shift_cols = df.columns[2:9]
    
    st.sidebar.success(f"कुल {len(shift_cols)} शिफ्ट्स लोड हुईं")
    target_shift = st.sidebar.selectbox("अपनी शिफ्ट चुनें:", shift_cols)

    if st.button("🔮 प्रेडिक्शन (00-99) शुरू करें"):
        clean_df = clean_and_prepare_data(df, shift_cols)
        
        if len(clean_df) < 10:
            st.error("डेटा बहुत कम है! कम से कम 10-15 पिछले नंबर्स होने चाहिए।")
        else:
            results = train_and_predict(clean_df, target_shift)
            
            st.subheader(f"📊 {target_shift} के लिए संभावित नंबर:")
            cols = st.columns(3)
            for i, res in enumerate(results):
                with cols[i]:
                    # यहाँ :02d सुनिश्चित करता है कि 5 की जगह 05 दिखे
                    st.markdown(f"""
                    <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; text-align:center;">
                        <h2 style="color:#ff4b4b;">RANK {i+1}</h2>
                        <h1 style="font-size:60px;">{res['number']:02d}</h1>
                        <p>संभावना: {res['prob']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
