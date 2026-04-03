import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time

# पेज कॉन्फ़िगरेशन
st.set_page_config(page_title="Number Prediction AI", layout="wide")

st.title("🎯 AI Number Guessing App (00-99)")
st.write("अपनी 3-5 साल की एक्सेल फाइल अपलोड करें और अगली संभावित संख्या का अनुमान लगाएं।")

# साइडबार - वॉलेट और लॉगिन (सिम्युलेटेड)
st.sidebar.header("👤 यूजर प्रोफाइल")
st.sidebar.info("लॉगिन: +91-XXXXX-XXXXX")
wallet_balance = st.sidebar.number_input("वॉलेट बैलेंस (Credits)", value=100)

# फाइल अपलोडर
uploaded_file = st.file_uploader("एक्सेल शीट अपलोड करें (CSV या XLSX)", type=['csv', 'xlsx'])

if uploaded_file:
    # डेटा लोड करना
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("डेटा सफलतापूर्वक लोड हो गया!")
        st.write("डेटा की झलक:", df.head())

        # मान लें कि आपकी संख्याएं 'Number' नामक कॉलम में हैं
        column_name = st.selectbox("उस कॉलम को चुनें जिसमें नंबर्स (00-99) हैं:", df.columns)

        if st.button("🔮 प्रेडिक्शन करें (10 Credits कटेंगे)"):
            if wallet_balance >= 10:
                with st.spinner('एआई डेटा पैटर्न को समझ रहा है...'):
                    
                    # डेटा प्री-प्रोसेसिंग (Lag Features बनाना)
                    data = df[column_name].values
                    X, y = [], []
                    # पिछले 5 नंबर्स के आधार पर अगला नंबर सीखना
                    window_size = 5 
                    for i in range(len(data) - window_size):
                        X.append(data[i:i + window_size])
                        y.append(data[i + window_size])
                    
                    X = np.array(X)
                    y = np.array(y)

                    # मॉडल ट्रेनिंग (Random Forest)
                    model = RandomForestClassifier(n_estimators=100)
                    model.fit(X, y)

                    # अगली संख्या की भविष्यवाणी
                    last_sequence = data[-window_size:].reshape(1, -1)
                    prediction = model.predict(last_sequence)[0]
                    
                    time.sleep(2) # थोड़ा प्रभाव डालने के लिए
                    st.sidebar.warning(f"नया वॉलेट बैलेंस: {wallet_balance - 10}")
                    
                    st.balloons()
                    st.markdown(f"""
                    <div style="text-align: center; border: 5px solid #4CAF50; padding: 20px; border-radius: 10px;">
                        <h1 style="color: #4CAF50;">अगली संभावित संख्या:</h1>
                        <h1 style="font-size: 100px; margin: 0;">{prediction:02d}</h1>
                    </div>
                    """, unsafe_allow_status=True)
            else:
                st.error("वॉलेट में पर्याप्त क्रेडिट नहीं है! कृपया रिचार्ज करें।")

    except Exception as e:
        st.error(f"त्रुटि: {e}. कृपया सुनिश्चित करें कि एक्सेल में सही डेटा है।")
        
