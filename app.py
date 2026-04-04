import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# पेज सेटअप
st.set_page_config(page_title="AI Predictor 00-99", layout="wide")

# डेटा को साफ करने का फंक्शन
def clean_data(df, shift_columns):
    temp_list = []
    for index, row in df.iterrows():
        try:
            # B कॉलम को तारीख मानकर (Index 1)
            dt = pd.to_datetime(row.iloc[1])
            day_val = dt.dayofweek
            
            for s_name in shift_columns:
                val = str(row[s_name]).strip()
                if val.isdigit():
                    n = int(val)
                    if 0 <= n <= 99:
                        temp_list.append({'day': day_val, 'shift': s_name, 'num': n})
        except:
            continue
    return pd.DataFrame(temp_list)

st.title("🎯 AI Number Guessing (00-99)")

# फाइल अपलोडर
uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx'])

if uploaded_file:
    try:
        data_df = pd.read_excel(uploaded_file)
        # C से I कॉलम (Index 2 से 8)
        s_cols = data_df.columns[2:9]
        
        selected_shift = st.selectbox("Select Shift", s_cols)
        
        if st.button("🔮 Predict"):
            clean_df = clean_data(data_df, s_cols)
            
            if len(clean_df) < 5:
                st.warning("Data is too small. Need more numbers.")
            else:
                # प्रेडिक्शन लॉजिक
                shifts = list(clean_df['shift'].unique())
                clean_df['s_id'] = clean_df['shift'].apply(lambda x: shifts.index(x))
                
                X = []
                y = []
                nums = clean_df['num'].values
                
                # पैटर्न के लिए लूप
                for i in range(3, len(nums)):
                    X.append(nums[i-3:i]) # पिछले 3 नंबर
                    y.append(nums[i])
                
                model = RandomForestClassifier(n_estimators=100)
                model.fit(np.array(X), np.array(y))
                
                # अगला नंबर अनुमान
                last_3 = nums[-3:].reshape(1, -1)
                probs = model.predict_proba(last_3)[0]
                top_3 = np.argsort(probs)[-3:][::-1]
                
                st.success("Predictions:")
                c1, c2, c3 = st.columns(3)
                for i, idx in enumerate(top_3):
                    with [c1, c2, c3][i]:
                        st.metric(f"Rank {i+1}", f"{model.classes_[idx]:02d}")
                st.balloons()
                
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload your Excel file to start.")
    
