import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Perceptron
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from streamlit_option_menu import option_menu
from collections import Counter

# Function to convert age to months
def convert_to_months(usia):
    usia = usia.replace('\xa0', ' ')
    parts = usia.split(' - ')
    try:
        tahun = int(parts[0].split(' ')[0])
        bulan = int(parts[1].split(' ')[0])
        hari = int(parts[2].split(' ')[0])
        total_bulan = (tahun * 12) + bulan
        return total_bulan
    except (IndexError, ValueError):
        return pd.NA

data = pd.read_csv('data_gizi.csv', encoding='latin1')

# Load the data
def load_data():
    df = data[['Usia Saat Ukur', 'Berat', 'Tinggi', 'BB/TB']]
    df.loc[:, 'Usia Saat Ukur'] = df['Usia Saat Ukur'].apply(convert_to_months)
    return df

df = load_data()
# Preprocessing steps
X = df.drop(columns=['BB/TB'])
y = df['BB/TB']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

smote_enn = SMOTEENN(random_state=42, smote=SMOTE(k_neighbors=1))
Xtrain_resampled, ytrain_resampled = smote_enn.fit_resample(X_train, y_train)
# Build and train the model
model = Perceptron(max_iter=1000, tol=1e-5, random_state=42)
model.fit(Xtrain_resampled, ytrain_resampled)
# Evaluate the model using cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=skf)
mean_accuracy = cv_scores.mean()
y_pred = model.predict(X_test)

# Menampilkan aplikasi Streamlit
def main():
    st.title("Klasifikasi Status Gizi Balita")
    page = option_menu(None, ["Data","Preprocessing", 'Modeling', "Implementasi"], 
        icons=['table', 'gear', 'diagram-3', 'play-circle'], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    page
    if page == "Data":
        st.header("Tentang Data")
        st.subheader("Data Awal")
        st.write(data)
        st.subheader("Data yang dipakai")
        st.write(df)
    # Preprocessing menu
    elif page == "Preprocessing":
        st.header("Preprocessing")
        st.subheader("Data Awal")
        st.write(df)
        st.subheader("Split Data Dan Normalisasi Data")
        # Membuat dua kolom
        col1, col2 = st.columns(2)

        # Menampilkan "Data Training" di kolom pertama
        with col1:
            st.write("Data Training")
            st.write(X_train)
            st.write(X_train.shape)

        # Menampilkan "Data Testing" di kolom kedua
        with col2:
            st.write("Data Testing")
            st.write(X_test)
            st.write(X_test.shape)

        st.subheader("Balencing data")
        st.write("Data yang dibalencing adalah data Training")
        col3, col4 = st.columns(2)
        with col3 :
            st.write("Sebelum di Balencing")
            counter1=Counter(y_train)
            st.write(counter1)
        with col4 :
            st.write("Sesudah di Balencing")
            counter=Counter(ytrain_resampled)
            st.write(counter)
    elif page == "Modeling":
        st.header("Model Yang digunakan adalah ANN Perceptron")
        st.write(f"Cross-validated Accuracy: {mean_accuracy}")
        st.subheader("Laporan Akurasi")
        # Membuat DataFrame dari laporan akurasi
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()
        # Menampilkan DataFrame sebagai tabel
        st.table(df_report[['precision', 'recall', 'f1-score', 'support']])
        
    elif page == "Implementasi":
        st.title("Implementasi")

        # Allow user to input data for prediction
        st.subheader("Input Data")
        berat = st.number_input("Berat", min_value=0.0, value=0.0)
        tinggi = st.number_input("Tinggi", min_value=0.0, value=0.0)
        usia_input = st.number_input("Usia(Bulan)", min_value=0.0, value=0.0)

        if st.button("Predict"):
            if usia_input == 0 or berat == 0 or tinggi == 0:
                st.write("Masukkan Data")
            else:
                user_data = scaler.transform([[usia_input, berat, tinggi]])
                prediction = model.predict(user_data)
                prediction_text = prediction[0]

                                # Apply color based on prediction value
                if prediction[0] == "Gizi Baik":
                    color = "green"
                elif prediction[0] == "Risiko Gizi Lebih":
                    color = "orange"
                else:
                    color = "red"
                
                st.markdown(f"Prediction: <h3 style='color: {color};'>{prediction[0]}</h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()