import pandas as pd
import numpy as np
import random 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def fetch_air_quality_indonesia(limit=1000, page=1):
    """
    Generate synthetic data untuk Indonesia - TANPA API
    """
    print("🔄 Generating synthetic air quality data for Indonesia...")
    return create_synthetic_dataset_indonesia(limit)

# PROVINSI DATA
PROVINSI_DATA = {
    'DKI Jakarta': (-6.2088, 106.8456),
    'Jawa Barat': (-6.9175, 107.6191),
    'Jawa Timur': (-7.2575, 112.7521),
    'Jawa Tengah': (-6.9914, 110.4208),
    'DI Yogyakarta': (-7.7956, 110.3695),
    'Banten': (-6.1147, 106.1597),
    'Sumatera Utara': (3.5952, 98.6728),
    'Kalimantan Timur': (1.4437, 116.7758),
    'Sulawesi Selatan': (-5.1474, 119.4101),
    'Papua': (-2.5489, 140.6493),
    'Maluku': (-3.6954, 128.1828),
}

# JAWA TIMUR KAB/KOTA
JAWA_TIMUR_DAERAH = {
    'Bangkalan': (-6.890, 113.390, {'pm25_factor': 0.9, 'pm10_factor': 1.0}),
    'Banyuwangi': (-8.210, 114.370, {'pm25_factor': 0.8, 'pm10_factor': 0.9}),
    'Blitar': (-7.990, 112.150, {'pm25_factor': 1.0, 'pm10_factor': 1.1}),
    'Bojonegoro': (-7.480, 111.390, {'pm25_factor': 1.05, 'pm10_factor': 1.15}),
    'Bondowoso': (-7.910, 113.830, {'pm25_factor': 0.95, 'pm10_factor': 1.05}),
    'Gresik': (-7.150, 112.650, {'pm25_factor': 1.2, 'pm10_factor': 1.3}),
    'Jember': (-8.170, 113.710, {'pm25_factor': 1.0, 'pm10_factor': 1.1}),
    'Jombang': (-7.550, 112.490, {'pm25_factor': 1.05, 'pm10_factor': 1.15}),
    'Kediri': (-7.820, 112.010, {'pm25_factor': 1.0, 'pm10_factor': 1.1}),
    'Lamongan': (-7.230, 112.910, {'pm25_factor': 1.1, 'pm10_factor': 1.2}),
    'Lumajang': (-8.180, 113.220, {'pm25_factor': 0.9, 'pm10_factor': 1.0}),
    'Madiun': (-7.630, 111.520, {'pm25_factor': 1.0, 'pm10_factor': 1.1}),
    'Magetan': (-7.530, 111.320, {'pm25_factor': 0.95, 'pm10_factor': 1.05}),
    'Malang': (-7.980, 112.630, {'pm25_factor': 1.05, 'pm10_factor': 1.15}),
    'Mojokerto': (-7.470, 112.430, {'pm25_factor': 1.1, 'pm10_factor': 1.2}),
    'Nganjuk': (-7.600, 111.900, {'pm25_factor': 1.0, 'pm10_factor': 1.1}),
    'Ngawi': (-7.400, 111.440, {'pm25_factor': 0.95, 'pm10_factor': 1.05}),
    'Pacitan': (-8.190, 111.080, {'pm25_factor': 0.8, 'pm10_factor': 0.9}),
    'Pamekasan': (-7.160, 113.910, {'pm25_factor': 0.9, 'pm10_factor': 1.0}),
    'Pasuruan': (-7.640, 112.910, {'pm25_factor': 1.1, 'pm10_factor': 1.2}),
    'Ponorogo': (-7.640, 111.480, {'pm25_factor': 0.95, 'pm10_factor': 1.05}),
    'Probolinggo': (-7.750, 113.210, {'pm25_factor': 1.0, 'pm10_factor': 1.1}),
    'Sampang': (-7.180, 113.240, {'pm25_factor': 0.9, 'pm10_factor': 1.0}),
    'Sidoarjo': (-7.450, 112.720, {'pm25_factor': 1.2, 'pm10_factor': 1.3}),
    'Situbondo': (-7.820, 114.010, {'pm25_factor': 0.85, 'pm10_factor': 0.95}),
    'Sumenep': (-7.020, 113.880, {'pm25_factor': 0.8, 'pm10_factor': 0.9}),
    'Trenggalek': (-8.070, 111.720, {'pm25_factor': 0.9, 'pm10_factor': 1.0}),
    'Tuban': (-6.880, 112.060, {'pm25_factor': 1.05, 'pm10_factor': 1.15}),
    'Tulungagung': (-8.060, 112.070, {'pm25_factor': 1.0, 'pm10_factor': 1.1}),
    # Kota
    'Surabaya': (-7.2575, 112.7521, {'pm25_factor': 1.15, 'pm10_factor': 1.25}),
    'Malang_Kota': (-7.976, 112.631, {'pm25_factor': 1.05, 'pm10_factor': 1.15}),
    'Batu': (-7.597, 112.525, {'pm25_factor': 0.9, 'pm10_factor': 1.0}),
    'Kediri_Kota': (-7.813, 112.015, {'pm25_factor': 1.0, 'pm10_factor': 1.1}),
    'Blitar_Kota': (-8.093, 112.150, {'pm25_factor': 0.95, 'pm10_factor': 1.05}),
    'Madiun_Kota': (-7.630, 111.520, {'pm25_factor': 1.0, 'pm10_factor': 1.1}),
    'Mojokerto_Kota': (-7.468, 112.434, {'pm25_factor': 1.1, 'pm10_factor': 1.2}),
    'Pasuruan_Kota': (-7.645, 112.907, {'pm25_factor': 1.1, 'pm10_factor': 1.2}),
    'Probolinggo_Kota': (-7.749, 113.214, {'pm25_factor': 1.0, 'pm10_factor': 1.1}),
}

# SURABAYA DISTRIK
SURABAYA_DISTRIK = {
    "Asemrowo": (-7.245, 112.700, {"pm25_factor": 1.1, "pm10_factor": 1.2}),
    "Benowo": (-7.242, 112.630, {"pm25_factor": 0.9, "pm10_factor": 1.0}),
    "Bubutan": (-7.246, 112.730, {"pm25_factor": 1.0, "pm10_factor": 1.1}),
    "Bulak": (-7.228, 112.790, {"pm25_factor": 0.95, "pm10_factor": 1.05}),
    "Dukuh Pakis": (-7.290, 112.710, {"pm25_factor": 1.05, "pm10_factor": 1.15}),
    "Gayungan": (-7.325, 112.720, {"pm25_factor": 1.0, "pm10_factor": 1.1}),
    "Genteng": (-7.260, 112.740, {"pm25_factor": 1.1, "pm10_factor": 1.2}),
    "Gubeng": (-7.275, 112.755, {"pm25_factor": 1.05, "pm10_factor": 1.15}),
    "Gunung Anyar": (-7.335, 112.770, {"pm25_factor": 0.95, "pm10_factor": 1.05}),
    "Jambangan": (-7.320, 112.730, {"pm25_factor": 1.0, "pm10_factor": 1.1}),
    "Karangpilang": (-7.340, 112.690, {"pm25_factor": 0.95, "pm10_factor": 1.05}),
    "Kenjeran": (-7.230, 112.800, {"pm25_factor": 1.1, "pm10_factor": 1.2}),
    "Krembangan": (-7.240, 112.730, {"pm25_factor": 1.05, "pm10_factor": 1.15}),
    "Lakarsantri": (-7.290, 112.630, {"pm25_factor": 0.9, "pm10_factor": 1.0}),
    "Mulyorejo": (-7.270, 112.790, {"pm25_factor": 1.0, "pm10_factor": 1.1}),
    "Pabean Cantikan": (-7.230, 112.730, {"pm25_factor": 1.05, "pm10_factor": 1.15}),
    "Pakal": (-7.250, 112.620, {"pm25_factor": 0.9, "pm10_factor": 1.0}),
    "Rungkut": (-7.330, 112.760, {"pm25_factor": 1.05, "pm10_factor": 1.15}),
    "Sambikerep": (-7.280, 112.670, {"pm25_factor": 0.9, "pm10_factor": 1.0}),
    "Sawahan": (-7.280, 112.720, {"pm25_factor": 1.05, "pm10_factor": 1.15}),
    "Semampir": (-7.220, 112.750, {"pm25_factor": 1.0, "pm10_factor": 1.1}),
    "Simokerto": (-7.255, 112.740, {"pm25_factor": 1.0, "pm10_factor": 1.1}),
    "Sukolilo": (-7.295, 112.770, {"pm25_factor": 1.05, "pm10_factor": 1.15}),
    "Sukomanunggal": (-7.275, 112.700, {"pm25_factor": 1.0, "pm10_factor": 1.1}),
    "Tambaksari": (-7.265, 112.760, {"pm25_factor": 1.1, "pm10_factor": 1.2}),
    "Tandes": (-7.265, 112.680, {"pm25_factor": 0.95, "pm10_factor": 1.05}),
    "Tegalsari": (-7.270, 112.730, {"pm25_factor": 1.0, "pm10_factor": 1.1}),
    "Tenggilis Mejoyo": (-7.320, 112.760, {"pm25_factor": 1.05, "pm10_factor": 1.15}),
    "Wiyung": (-7.315, 112.690, {"pm25_factor": 0.95, "pm10_factor": 1.05}),
    "Wonocolo": (-7.315, 112.740, {"pm25_factor": 1.0, "pm10_factor": 1.1}),
    "Wonokromo": (-7.295, 112.735, {"pm25_factor": 1.05, "pm10_factor": 1.15}),
}

def _label_from_pm25(pm25):
    """Convert PM2.5 value to label"""
    if pm25 < 15:
        return 'Aman'
    elif pm25 < 35:
        return 'Waspada'
    elif pm25 < 55:
        return 'Tidak Aman bagi Rentan'
    else:
        return 'Tidak Aman'

def _aktivitas_rekomendasi(label):
    """Get activity recommendation based on label"""
    if label == "Aman":
        return "Semua aktivitas luar ruangan bebas."
    elif label == "Waspada":
        return "Aktivitas boleh, tapi jangan terlalu berat."
    elif label == "Tidak Aman bagi Rentan":
        return "Kelompok rentan sebaiknya batasi aktivitas luar."
    elif label == "Tidak Aman":
        return "Hindari aktivitas luar ruangan."
    return "Label tidak dikenali."

def create_synthetic_dataset_indonesia(n_samples=2000, random_state=42, fokus_jatim=False):
    """Create synthetic air quality dataset for Indonesia"""
    random.seed(random_state)
    np.random.seed(random_state)

    rows = []
    provs = list(PROVINSI_DATA.keys())

    for _ in range(n_samples):
        prov = "Jawa Timur" if fokus_jatim else random.choice(provs)

        base_pm25 = np.clip(np.random.normal(30, 15), 2, 200)
        base_pm10 = np.clip(base_pm25 * np.random.uniform(2.5, 4.0), 10, 500)
        co = np.clip(np.random.normal(10, 5), 0.1, 50)
        suhu = np.clip(np.random.normal(28, 2), 18, 38)
        kelembaban = np.clip(np.random.normal(75, 8), 30, 100)

        kab = "Umum"
        distrik = "Umum"

        if prov == "Jawa Timur":
            kab = random.choice(list(JAWA_TIMUR_DAERAH.keys()))
            factor = JAWA_TIMUR_DAERAH[kab][2]
            base_pm25 *= factor.get("pm25_factor", 1.0)
            base_pm10 *= factor.get("pm10_factor", 1.0)

            if kab == "Surabaya":
                distrik = random.choice(list(SURABAYA_DISTRIK.keys()))
                f2 = SURABAYA_DISTRIK[distrik][2]
                base_pm25 *= f2.get("pm25_factor", 1.0)
                base_pm10 *= f2.get("pm10_factor", 1.0)

        label = _label_from_pm25(base_pm25)
        rekomendasi = _aktivitas_rekomendasi(label)

        rows.append({
            "PM2.5": round(float(base_pm25), 2),
            "PM10": round(float(base_pm10), 2),
            "CO": round(float(co), 2),
            "Suhu": round(float(suhu), 2),
            "Kelembaban": round(float(kelembaban), 2),
            "Provinsi": prov,
            "Kabupaten": kab,
            "Distrik": distrik,
            "Label": label,
            "Rekomendasi": rekomendasi
        })

    return pd.DataFrame(rows)

def prepare_data(df, test_size=0.2, random_state=42):
    """Prepare data for training"""
    numeric_features = ['PM2.5', 'PM10', 'CO', 'Suhu', 'Kelembaban']
    
    # Ensure all numeric columns exist
    for col in numeric_features:
        if col not in df.columns:
            df[col] = 0.0

    y = df['Label'].astype(str).values
    X_num = df[numeric_features].astype(float).values
    
    # One-hot encode provinces
    prov_dummies = pd.get_dummies(df['Provinsi'].astype(str), prefix='Prov')
    prov_columns = list(prov_dummies.columns)
    X_full = np.hstack([X_num, prov_dummies.values])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale numeric features
    scaler = StandardScaler()
    scaler.fit(X_train[:, :len(numeric_features)])

    X_train_scaled = np.hstack([scaler.transform(X_train[:, :len(numeric_features)]), 
                                X_train[:, len(numeric_features):]])
    X_test_scaled = np.hstack([scaler.transform(X_test[:, :len(numeric_features)]), 
                               X_test[:, len(numeric_features):]])

    preprocessor_info = {
        'scaler': scaler,
        'provinsi_columns': prov_columns,
        'numeric_features': numeric_features
    }

    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor_info

if __name__ == "__main__":
    df = create_synthetic_dataset_indonesia(1000, fokus_jatim=True)
    print(df.head())
    X_train, X_test, y_train, y_test, preproc = prepare_data(df)
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)