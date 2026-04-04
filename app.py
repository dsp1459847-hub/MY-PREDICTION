import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- 1. डेटा क्लीनिंग फंक्शन (A1-I1 Structure के लिए) ---
def clean_and_prepare_data(df, shift_columns):
    all_data = []
    # B1 कॉलम को डेट (Date) मानकर चलना
    for index, row in df.iterrows():
        try:
            if pd.isna(row.iloc[1]): continue 
            
            current_date = pd.to_datetime(row.iloc[1])
            day_of_week = current_date.dayofweek
            
            for shift_name in shift_columns:
                val = str(row[shift_name]).strip()
                # सिर्फ 0-99 के बीच के नंबर्स लेना, XX या टेक्स्ट को छोड़ना
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
    
    clean_df = pd.DataFrame(all_data)
    if not clean_df.empty:
        clean_df = clean_df.sort_values(['date', 'shift'])
    return clean_df

# --- 2. AI ट्रेनिंग और प्रेडिक्शन फंक्शन ---
def train_and_predict(clean_df, target_shift):
    shift_list = list(clean_df['shift'].unique())
    clean_df['shift_id'] = clean_df['shift'].apply(lambda x: shift_list.index(x))
    
    window = 5 # पिछले 5 नंबर्स का पैटर्न
    X, y = [], []
    nums = clean_df['number'].values
    days = clean_df['day_of_week'].values
    shifts = clean_df['shift_id'].values
    
    for i in range(window, len(clean_df)):
        features = list(nums[i-window:i]) + [days[i], shifts[i]]
        X.append(features)
        y.append(nums[i])
        
    # AI मॉडल (Random Forest)
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(np.array(X), np.array(y))
    
    # अगली प्रेडिक्शन के लिए इनपुट
    target_shift_id = shift_list.index(target_shift)
    next_day = days[-1] 
    last_features = list(nums[-window:]) + [next_day, target_shift_id]
    
    # टॉप 3 रिजल्ट्स निकालना
    probs = model.predict_proba([last_features])[0]
    top_3_indices = np.argsort(probs)[-3:][::-1]
    
    results = []
    for idx in top_3_indices:
        results.append({
            'number': model.classes_[idx],
            'prob': probs[idx] * 100
        })
    return results

# --- 3. यूजर इंटरफेस (Streamlit UI) ---
st.set_page_config(page_title="AI Predictor 00-99", layout="wide")

st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🎯 Predictor (00-99)</h1>", unsafe_allow_html=True)
st.write("---")

uploaded_file = st.file_uploader("अपनी Excel फाइल अपलोड करें", type=['xlsx'])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        # शिफ्ट कॉलम्स पहचानना (C1 से I1)
        shift_cols = df.columns[2:9]
        
        col1, col2 = st.columns([2, 1])
        with col1:
            target_shift = st.selectbox("किस शिफ्ट के लिए प्रेडिक्शन चाहिए?", shift_cols)
        
        if st.button("🔮 Predict Next Number"):
            with st.spinner('एआई पुराने पैटर्न का विश्लेषण कर रहा है...'):
                clean_df = clean_and_prepare_data(df, shift_cols)
                
                if clean_df.empty:
                    st.error("एक्सेल में कोई वैध नंबर (00-99) नहीं मिला!")
                else:
                    predictions = train_and_predict(clean_df, target_shift)
                    
                    st.success("प्रेडिक्शन तैयार है!")
                    st.write(f"### {target_shift} के लिए टॉप 3 सुझाव:")
                    
                    # डिस्प्ले कार्ड्स
                    res_cols = st.columns(3)
                    for i, res in enumerate(predictions):
                        with res_cols[i]:
                            st.markdown(f"""
                            <div style="background-color:#1e1e1e; padding:30px; border-radius:15px; text-align:center; border: 2px solid #ff4b4b; box-shadow: 5px 5px 15px rgba(0,0,0,0.3);">
                                <h2 style="color:#ff4b4b; margin:0;">Rank {i+1}</h2>
                                <h1 style="font-size:70px; color:white; margin:10px 0;">{res['number']:02d}</h1>
                                <p style="color:#00ff00; font-size:20px; font-weight:bold;">{res['prob']:.1f}% संभावना</p>
                            </div>
                            """, unsafe_allow_html=True)
                    st.balloons()
    except Exception as e:
        st.error(f"फाइल पढ़ने में त्रुटि हुई: {e}")

else:
    st.info("कृपया ऊपर दी गई जगह में अपनी एक्सेल फाइल अपलोड करें।")
