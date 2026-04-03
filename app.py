import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# डेटा तैयार करने का एडवांस फंक्शन
def prepare_advanced_data(df, date_col, num_col, shift_col):
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([date_col, shift_col]) # तारीख और शिफ्ट के अनुसार सॉर्ट
    
    # फीचर्स: दिन, शिफ्ट, और पिछले 3 नंबर्स
    df['day_of_week'] = df[date_col].dt.dayofweek
    # शिफ्ट को नंबर में बदलना (e.g., Morning=0, Evening=1)
    df['shift_id'] = pd.factorize(df[shift_col])[0]
    
    window = 3
    X, y = [], []
    nums = df[num_col].values
    shifts = df['shift_id'].values
    days = df['day_of_week'].values
    
    for i in range(window, len(df)):
        # फीचर: [पिछला नंबर 1, पिछला नंबर 2, पिछला नंबर 3, आज का दिन, आज की शिफ्ट]
        features = list(nums[i-window:i]) + [days[i], shifts[i]]
        X.append(features)
        y.append(nums[i])
        
    return np.array(X), np.array(y), nums, days, df[shift_col].unique()

st.title("🎯 AI Multi-Shift Predictor")

uploaded_file = st.file_uploader("अपनी Excel/CSV फाइल अपलोड करें", type=['xlsx', 'csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    
    col1, col2, col3 = st.columns(3)
    date_col = col1.selectbox("Date Column:", df.columns)
    shift_col = col2.selectbox("Shift/Time Column:", df.columns)
    num_col = col3.selectbox("Number Column:", df.columns)
    
    target_shift = st.selectbox("किस शिफ्ट के लिए प्रेडिक्शन चाहिए?", df[shift_col].unique())

    if st.button("🔮 Analyze Shift Patterns"):
        X, y, nums, days, shift_names = prepare_advanced_data(df, date_col, num_col, shift_col)
        shift_id = list(shift_names).index(target_shift)
        
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X, y)
        
        # अगली शिफ्ट के लिए डेटा (कल का दिन और चुनी हुई शिफ्ट)
        next_day = (days[-1] + 1) % 7
        last_features = list(nums[-3:]) + [next_day, shift_id]
        
        probs = model.predict_proba([last_features])[0]
        top_3_indices = np.argsort(probs)[-3:][::-1]
        
        st.divider()
        st.subheader(f"✅ {target_shift} के लिए टॉप 3 नंबर:")
        
        res_cols = st.columns(3)
        for i in range(3):
            num = model.classes_[top_3_indices[i]]
            prob = probs[top_3_indices[i]] * 100
            res_cols[i].metric(f"Rank {i+1}", f"{num:02d}", f"{prob:.1f}% Match")

        st.info(f"नोट: यह प्रेडिक्शन विशेष रूप से '{target_shift}' के ऐतिहासिक डेटा पर आधारित है।")
      
