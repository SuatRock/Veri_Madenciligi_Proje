import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


"""VERININ ANLASILMASI"""


# Ciktilarin Terminaldeki Gorsel Ayarlarini DUzenleme
pd.set_option("display.max_columns", 35)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 150)


# Veri Setinin import edilmesi
df = pd.read_csv(
    r"C:\\Users\\TUF\\OneDrive\\Masaüstü\\Proje\\WA_Fn-UseC_-HR-Employee-Attritionn.csv"
)


# Verinin Anlasilmasi Icin Fonksiyonun Olusturulmasi
def check_df(dataframe, NOC=3, Depnted_Value=df["Attrition"]):

    print("----SHAPE----")
    print(dataframe.shape)

    print("----TYPES----")
    print(dataframe.dtypes)

    print("----HEAD----")
    print(dataframe.head(NOC))

    print("----TAİL----")
    print(dataframe.tail(NOC))

    print("----UNİQ----")
    print(dataframe.nunique())

    print("----DUP----")
    print(dataframe.duplicated().sum())

    print("----NA----")
    print(dataframe.isnull().sum())

    print("----QUANTİLES----")
    print(dataframe.describe().T)

    print("-----BALANCE-----")
    print(Depnted_Value.value_counts())


check_df(df)


""""VERI ONISLEME"""


# Unique Degerleri "1" Olan Sutunlarin Atilmasi Ve Kaydedilmesi
df.drop(
    ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"], axis=1, inplace=True
)
df1 = df.copy()


# Object Degerlerin Sayisal Degelere Donusturulmesi
label_encoder = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df1[col] = label_encoder.fit_transform(df1[col])


# Bagimsiz ve Bagimli Degiskenler Olarak Veriyi Ayirma
x = df1.drop("Attrition", axis=1)
y = df1["Attrition"]


# Verinin Test ve Tranning Verisi Olarak Ayarlanmasi
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=100
)


# Veri Setinin Standardize Edilmesi
scaler = StandardScaler()
sca_x_train = scaler.fit_transform(x_train)
sca_x_test = scaler.transform(x_test)


"""MODELLEME(MODELIN KURULMASI)"""


# Ikili Lojistik Regresyonun Kurulmasi
lg_model = LogisticRegression(C=0.02, max_iter=25000)
lg_model.fit(sca_x_train, y_train)


"""MODELIN DEGERLENDIRILMESI"""


# Modelin Egtim Verisi Uzerindeki Skoru
print(lg_model.score(sca_x_train, y_train))


# Confusion matrix olusturulmasi
tahmin = lg_model.predict(sca_x_test)
cm = confusion_matrix(y_test, tahmin)
print(cm)


# Confusion_matrix Gorsellestirilmesi
disp = ConfusionMatrixDisplay(display_labels=lg_model.classes_, confusion_matrix=cm)
disp.plot()
plt.show()


# Recall - Accuracy - Preccision Degerelerinin Gorulmesi
print((classification_report(y_test, tahmin)))


# Coef Degerlerini OLusturma
coef = pd.Series(index=x.columns, data=lg_model.coef_[0])


# Coef Siralama
coef_soarted = coef.sort_values()
print(coef_soarted)


# Coef Negatif Pozitif Olarak Ikiye Bolme
pozitif_coef = coef_soarted[coef_soarted >= 0.10] * 100
negatif_coef = coef_soarted[coef_soarted <= -0.15] * 100


print(negatif_coef)
print(pozitif_coef.sort_values(ascending=False))


# Coef Grafik Olarak Gorsellestirme
sns.barplot(
    x=pozitif_coef.index,
    y=pozitif_coef.values,
    hue=pozitif_coef,
    palette="bright",
)
plt.show()


sns.barplot(
    x=negatif_coef.index,
    y=negatif_coef.values,
    hue=negatif_coef,
    palette="bright",
)
plt.show()


"""MODEDELIN KULLANILMASI"""


# Veri seti Uzerinden Gercek Veri OrneGi Secimi
print(df1.tail(2))
print(df1.head(2))


# Ornek Verileri Olusturma
t1 = [
    [
        38.94,
        2,
        467.5433,
        2,
        22.28,
        3.8,
        3,
        2.5,
        1,
        40.93,
        2.0,
        3.0,
        6,
        1.0,
        0,
        10813.0,
        15511.0,
        4.1,
        1,
        12.5,
        3.0,
        2.38,
        0.32,
        17.45,
        2.9,
        1.44,
        3.84,
        1.7,
        4.17,
        5.5,
    ]
]


t2 = [
    [
        25.0,
        2,
        1102.0,
        2,
        5.0,
        2.0,
        1,
        1.0,
        0,
        94.0,
        2.0,
        2.0,
        6,
        3.0,
        2,
        4993.0,
        19479.0,
        8.0,
        1,
        11.0,
        2.0,
        1.0,
        0.0,
        8.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        2.0,
    ]
]


t3 = [
    [
        70.0,
        2,
        1102.0,
        2,
        1.0,
        4.0,
        1,
        1.0,
        0,
        61.0,
        4.0,
        3.0,
        6,
        4.0,
        2,
        9993.0,
        19479.0,
        8.0,
        0,
        11.0,
        4.0,
        3.0,
        2.0,
        11.0,
        1.0,
        3.0,
        10.0,
        4.0,
        0.0,
        7.0,
    ]
]


# Verileri Test Etmek Icin Fonksiyon Olusturma ve Kullanma
def predict_sample(smp1, smp2, smp3):

    print("----SAMPLE 1----")
    smp1_sca = scaler.transform(smp1)
    pre1 = lg_model.predict(smp1_sca)
    pre1_pro = lg_model.predict_proba(smp1_sca) * 100
    print(pre1, "------", pre1_pro)

    print("----SAMPLE 2----")
    smp2_sca = scaler.transform(smp2)
    pre2 = lg_model.predict(smp2_sca)
    pre2_pro = lg_model.predict_proba(smp2_sca) * 100
    print(pre2, "------", pre2_pro)

    print("----SAMPLE 3----")
    smp3_sca = scaler.transform(smp3)
    pre3 = lg_model.predict(smp3_sca)
    pre3_pro = lg_model.predict_proba(smp3_sca) * 100
    print(pre3, "------", pre3_pro)


predict_sample(t1, t2, t3)
