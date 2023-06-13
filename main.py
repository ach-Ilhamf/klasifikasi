import streamlit as st

import sklearn
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


import pickle

from sklearn import metrics

st.set_page_config(
    page_title="Prediksi Penyakit Ginjal Kronis"
)

st.title('Prediksi Penyakit Ginjal Kronis')
st.write("""
Nama : Achmad Ilham Firmansyah
""")
st.write("""
NIM : 210411100127
""")

tab1, tab2, tab3, tab4 = st.tabs(["Data", "Preprocessing", "Modelling", "Implementation"])

with tab1:
    st.write("""
    <h4>Data </h4>
    <br>
    """, unsafe_allow_html=True)


    st.write("""<h4> Aplikasi Untuk Memprediksi Penyakit Ginjal Kronis <h4>""", unsafe_allow_html=True)

    st.markdown("""
    Dataset penyakit ginjal kronis diambil dari link kaggle berikut :
    <a href="https://www.kaggle.com/datasets/abhia1999/chronic-kidney-disease"> https://www.kaggle.com/datasets/abhia1999/chronic-kidney-disease</a>
    """, unsafe_allow_html=True)

    st.markdown("""
    Link Repository Github
    https://raw.githubusercontent.com/ach-Ilhamf/data_csv/main/penyakit_ginjal.csv
    """, unsafe_allow_html=True)
    

    st.write("Tipe data yang terdapat di dalam dataset yaitu tipe data biner dan tip data numerik")

    st.write('Dataset ini berisi tentang klasifikasi penyakit ginjal kronis')
    df = pd.read_csv("https://raw.githubusercontent.com/ach-Ilhamf/data_csv/main/penyakit_ginjal.csv")
    st.write("Dataset penyakit ginjal kronis : ")
    st.write(df)
    st.write("Penjelasan fitur-fitur yang ada")

    st.write("""
    <ol>
    <li>Bp : Blood pressure (tekanan darah)</li>
    <li>Sg : Spesific Gravity (berat jenis) memberikan informasi untuk mengindikasikan fungsi ginjal yang normal atau adanya masalah ginjal.</li>
    <li>Al : Albumin, pengukuran albuminuria digunakan sebagai indikator kerusakan ginjal dan kehilangan fungsi penyaringan ginjal yang terjadi pada penyakit ginjal kronis.</li>
    <li>Su : Sugar </li>
    <li>Rbc : Red blood cell (sel darah merah) pemeriksaan sel darah merah dapat memberikan informasi penting tentang kondisi kesehatan ginjal dan dapat menjadi petunjuk adanya komplikasi terkait. </li>
    <li>Bu: Blood urea dapat digunakan sebagai salah satu parameter dalam klasifikasi penyakit ginjal kronis</li>
    <li>Sc : Serum Creatinine (Kreatinin serum) Kadar kreatinin serum yang tinggi mengindikasikan penurunan fungsi ginjal dan menunjukkan adanya kerusakan ginjal.</li>
    <li>Sod : Sodium (natrium) kadar natrium serum dapat mencerminkan keseimbangan elektrolit dalam tubuh dan fungsi ginjal yang terganggu.</li>
    <li>Pot :  Pottasium (kalium) kadar kalium serum mencerminkan keseimbangan elektrolit dalam tubuh dan fungsi ginjal yang terganggu.</li>
    <li>Hemo : Hemoglobin penurunan fungsi ginjal dapat menyebabkan anemia, yaitu penurunan jumlah sel darah merah atau kadar hemoglobin yang rendah.</li>
    <li>Wbcc : White Blood Cell Count (jumlah sel darah putih) peningkatan jumlah sel darah putih dapat terjadi sebagai respons terhadap kerusakan ginjal atau infeksi yang terkait..</li>
    <li>Rbcc : Red Blood Cell Count (jumlah sel darah merah) penurunan sel darah merah dapat menjadi tanda perburukan kondisi ginjal.</li>
    <li>Htn : Hypertension (tekanan darah tinggi) dapat menyebabkan kerusakan pembuluh darah ginjal dan memperburuk fungsi ginjal yang sudah terganggu. /li>
    <li>Class: hasil diagnosa penyakit ginjal kronis, 0 untuk terdiagnosa positif terkena penyakit ginjal kronis, dan 1 untuk negatif terkena penyakit ginjal kronis.</li>
    </ol>
    """,unsafe_allow_html=True)

with tab2:
    st.write("""
    <h4>Preprocessing Data</h4>
    <br>
    """, unsafe_allow_html=True)
    
    scaler = st.radio(
    "Metode normalisasi data",
    ('Min max scaler','Tanpa scaler'))
    if scaler == 'Tanpa scaler':
        st.write("Dataset Tanpa Preprocessing : ")
        df_new=df
    elif scaler == 'Min max scaler':
        st.write("Dataset setelah Preprocessing dengan MinMax Scaler: ")
        scaler = MinMaxScaler()
        df_for_scaler = pd.DataFrame(df, columns = ['Bp','Sg','Al','Su','Bu','Sc','Sod','Pot','Hemo','Wbcc','Rbcc'])
        df_for_scaler = scaler.fit_transform(df_for_scaler)
        df_for_scaler = pd.DataFrame(df_for_scaler,columns = ['Bp','Sg','Al','Su','Bu','Sc','Sod','Pot','Hemo','Wbcc','Rbcc'])
        df_drop_column_for_minmaxscaler=df.drop(['Bp','Sg','Al','Su','Bu','Sc','Sod','Pot','Hemo','Wbcc','Rbcc'], axis=1)
        df_new = pd.concat([df_for_scaler,df_drop_column_for_minmaxscaler], axis=1)
    st.write(df_new)

with tab3:
    st.write("""
    <h4>Modelling</h4>
    <br>
    """, unsafe_allow_html=True)

    nb = st.checkbox("Naive Bayes")  # Checkbox for Naive Bayes
    knn = st.checkbox("KNN")  # Checkbox for KNN
    ds = st.checkbox("Decision Tree")  # Checkbox for Decision Tree
    mlp = st.checkbox("MLP")  # Checkbox for MLP

    # Splitting the data into features and target variable
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Splitting the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = []  # List to store selected models

    if nb:
        models.append(('Naive Bayes', GaussianNB()))
    if knn:
        models.append(('KNN', KNeighborsClassifier()))
    if ds:
        models.append(('Decision Tree', DecisionTreeClassifier()))
    if mlp:
        models.append(('MLP', MLPClassifier()))

    if len(models) == 0:
        st.warning("Please select at least one model.")

    else:
        accuracy_scores = []  # List to store accuracy scores

        st.write("<h6>Accuracy Scores:</h6>", unsafe_allow_html=True)
        st.write("<table><tr><th>Model</th><th>Accuracy</th></tr>", unsafe_allow_html=True)

        for model_name, model in models:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)
            st.write("<tr><td>{}</td><td>{:.2f}</td></tr>".format(model_name, accuracy), unsafe_allow_html=True)

        st.write("</table>", unsafe_allow_html=True)

        # Displaying the table of test labels and predicted labels
        st.write("<h6>Test Labels and Predicted Labels:</h6>", unsafe_allow_html=True)
        labels_df = pd.DataFrame({'Test Labels': y_test, 'Predicted Labels': y_pred})
        st.write(labels_df)


# Define the decision tree classifier model
model = DecisionTreeClassifier()

# Fit the model to the training data
model.fit(X_train, y_train)

# Save the decision tree model as a pickle file
filename = 'decision_tree.pkl'
pickle.dump(model, open(filename, 'wb'))

with tab4:
    st.write("""
    <br>
    """, unsafe_allow_html=True)
    st.write("""
    <h4>Implementasi</h4>
    <br>
    """, unsafe_allow_html=True)
    X=df_new.iloc[:,0:13].values
    y=df_new.iloc[:,13].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=42)

    bp=st.number_input("Tekanan darah mm/hg : ")
    sg=st.number_input("Berat jenis : ")

    al=st.selectbox(
        'Albumin',
        ('1','2','3','4','5')
    )
    if al=='1':
        al=1.0
    elif al=='2':
        al=2.0
    elif al=='3':
        al=3.0
    elif al=='4':
        al=4.0
    elif al=='5':
        al=5.0
    su=st.selectbox(
        'Kadar gula',
        ('1','2','3','4','5')
    )
    if su=='1':
        su=1.0
    elif su=='2':
        su=2.0
    elif su=='3':
        su=3.0
    elif su=='4':
        su=4.0
    elif su=='5':
        su=5.0
    rbc=st.selectbox(
        'Sel darah merah',
        ('0', '1')
    )
    if rbc =='0':
        rbc=0.0
    elif rbc=='1':
        rbc=1.0
    bu=st.number_input("Blood urea mM : ")
    sc=st.number_input("Kreatinin serum mg/dl : ")
    sod=st.number_input("Kadar Sodium : ")
    pot=st.number_input("Kadar kalium : ")
    hemo=st.number_input("Kadar hemoglobin g/dL : ")
    wbcc= st.number_input("Jumlah sel darah putih sel/mm^3 : ")
    rbcc= st.number_input("Jumlah sel darah merah jutasel/mm^3 : ")
    htn=st.selectbox(
        'Tekanan darah tinggi',
        ('Tidak', 'Ya')
    )
    if htn =='Tidak':
        htn=0.0
    elif htn=='Ya':
        htn=1.0

    algoritma2 = st.selectbox(
        'Model Terbaik: ',
        ('Decision Tree','Decision Tree')
    )
    model2 = DecisionTreeClassifier()
    filename2 = 'decision_tree.pkl'

    algoritma = st.selectbox(
        'pilih model klasifikasi lain :',
        ('KNN','Naive Bayes', 'MLP')
    )
    prediksi=st.button("Diagnosis")
    if prediksi:
        if algoritma=='KNN':
            model = KNeighborsClassifier(n_neighbors=3)
            filename='knn.pkl'
        elif algoritma=='Naive Bayes':
            model = GaussianNB()
            filename='gaussian.pkl'
        elif algoritma=='MLP':
            model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
            filename='mlp.pkl'
        
        model2.fit(X_train, y_train)
        Y_pred2 = model2.predict(X_test) 

        score2=metrics.accuracy_score(y_test,Y_pred2)

        loaded_model2 = pickle.load(open(filename2, 'rb'))

        model.fit(X_train, y_train)
        Y_pred = model.predict(X_test) 

        score=metrics.accuracy_score(y_test,Y_pred)

        loaded_model = pickle.load(open(filename, 'rb'))
        if scaler == 'Tanpa Scaler':
            dataArray = [bp,sg,al,su,rbc,bu,sc,sod,pot,hemo,wbcc,rbcc,htn]
        else:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            bp_proceced = (bp - df['Bp'].mean()) / df['Bp'].std()
            sg_proceced = (sg - df['Sg'].mean()) / df['Sg'].std()
            al_proceced = (al - df['Al'].mean()) / df['Al'].std()
            su_proceced = (su - df['Su'].mean()) / df['Su'].std()
            bu_proceced = (bu - df['Bu'].mean()) / df['Bu'].std()
            sc_proceced = (sc - df['Sc'].mean()) / df['Sc'].std()
            sod_proceced = (sod - df['Sod'].mean()) / df['Sod'].std()
            pot_proceced = (pot - df['Pot'].mean()) / df['Pot'].std()
            hemo_proceced = (hemo - df['Hemo'].mean()) / df['Hemo'].std()
            wbcc_proceced = (wbcc - df['Wbcc'].mean()) / df['Wbcc'].std()
            rbcc_proceced = (rbcc - df['Rbcc'].mean()) / df['Rbcc'].std()

            dataArray = [
                bp_proceced, sg_proceced, al_proceced, su_proceced, bu_proceced, sc_proceced,
                sod_proceced, pot_proceced, hemo_proceced, wbcc_proceced, rbcc_proceced, rbc, htn
            ]

        pred = loaded_model.predict([dataArray])
        pred2 = loaded_model2.predict([dataArray])

        st.write('--------')
        st.write('Hasil dengan Decision Tree :')
        if int(pred2[0])==1:
            st.success(f"Hasil Prediksi : Tidak memiliki penyakit ginjal kronis")
        elif int(pred2[0])==0:
            st.error(f"Hasil Prediksi : Memiliki penyakit penyakit ginjal kronis")

        st.write(f"akurasi : {score2}")
        st.write('--------')
        st.write('Hasil dengan ',{algoritma},' :')
        if int(pred[0])==0:
            st.success(f"Hasil Prediksi : Tidak memiliki penyakit ginjal kronis")
        elif int(pred[0])==1:
            st.error(f"Hasil Prediksi : Memiliki penyakit ginjal kronis")

        st.write(f"akurasi : {score}")
