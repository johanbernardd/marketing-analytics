import streamlit as st
import joblib
import numpy as np
import xgboost

model = joblib.load('D:\\johanbernardd\\uni\\COURSE\\Self-Project\\marketing-analytics\\xg_boost_model.pkl')

st.title("Marketing Analytics-Campaign Response Prediction")
st.write("Author: Johanes Paulus Bernard Purek")

st.subheader("Customer Profile")
LiveWith = st.selectbox("Is the customer married or living together?", [0, 1], index=1)
Education = st.selectbox("Does the customer have a degree?", [0, 1], index=1)
Income = st.number_input("Yearly Household Income", min_value=0, max_value=113734, value=51687)
Kidhome = st.number_input("Number of children", min_value=0, max_value=4, value=1)
Teenhome = st.number_input("Number of teenagers", min_value=0, max_value=4, value=1)
Customer_Days = st.number_input("Days since registration", min_value=0, max_value=2858, value=2511)
Recency = st.number_input("Days since last purchase", min_value=0, max_value=99, value=49)
Dependents = st.number_input("Number of dependents", min_value=0, value=2)
Has_Dependent = st.selectbox("Has at least 1 dependent?", [0, 1], index=1)

st.subheader("Product Preferences")
MntWines = st.number_input("Spent on wine", min_value=0, max_value=1493, value=306)
MntFruits = st.number_input("Spent on fruits", min_value=0, max_value=199, value=26)
MntMeatProducts = st.number_input("Spent on meat", min_value=0, max_value=1725, value=165)
MntFishProducts = st.number_input("Spent on fish", min_value=0, max_value=259, value=38)
MntSweetProducts = st.number_input("Spent on sweets", min_value=0, max_value=262, value=27)
MntRegularProds = st.number_input("Spent on regular products", min_value=0, max_value=2458, value=519)
MntGoldProds = st.number_input("Spent on gold", min_value=0, max_value=321, value=44)
MntTotal = st.number_input("Total amount spent", min_value=0, max_value=2491, value=563)

st.subheader("Campaign Responses")
AcceptedCmp1 = st.selectbox("Accepted 1st campaign?", [0, 1], index=1)
AcceptedCmp3 = st.selectbox("Accepted 3rd campaign?", [0, 1], index=1)
AcceptedCmp5 = st.selectbox("Accepted 5th campaign?", [0, 1], index=1)
AcceptedCmpOverall = st.slider("Number of accepted campaigns (max 3)", 0, 3, 3)
IsRetented = st.selectbox("Accepted more than 1 campaign?", [0, 1], index=1)

st.subheader("Channel Performance")
NumWebPurchases = st.number_input("Web purchases", min_value=0, max_value=27, value=4)
NumCatalogPurchases = st.number_input("Catalog purchases", min_value=0, max_value=28, value=3)
NumStorePurchases = st.number_input("Store purchases", min_value=0, max_value=13, value=6)
NumWebVisitsMonth = st.number_input("Website visits last month", min_value=0, max_value=20, value=5)
NumTotalPurchases = st.number_input("Total purchases", min_value=0, max_value=68, value=13)

if st.button("Predict Response"):
    input_data = np.array([[LiveWith, Education, Income, Kidhome, Teenhome, Customer_Days,
                            Recency, Dependents, Has_Dependent, MntWines, MntFruits, MntMeatProducts,
                            MntFishProducts, MntSweetProducts, MntRegularProds, MntGoldProds, MntTotal,
                            AcceptedCmp1, AcceptedCmp3, AcceptedCmp5, AcceptedCmpOverall, IsRetented,
                            NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth,
                            NumTotalPurchases]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    if prediction == 1:
        st.markdown('<div style="background-color:#d4edda;padding:20px;border-radius:10px;"><h3 style="color:#155724;">✅ The customer is likely to ACCEPT the offer!</h3></div>',
        unsafe_allow_html=True
    )
    else:
        st.markdown('<div style="background-color:#f8d7da;padding:20px;border-radius:10px;"><h3 style="color:#721c24;">❌ The customer is NOT likely to accept the offer.</h3></div>',
        unsafe_allow_html=True
    )
    st.caption(f"Model Confidence (Accept Probability): {probability:.2%}")
