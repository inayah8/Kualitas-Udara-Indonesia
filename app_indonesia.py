import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
import folium
from streamlit_folium import folium_static
import random

# Data lokasi dan faktor polusi
PROVINSI_DATA = {
    'DKI Jakarta': (-6.2088, 106.8456, {'pm25_factor': 1.3, 'pm10_factor': 1.4}),
    'Jawa Barat': (-6.9175, 107.6191, {'pm25_factor': 1.1, 'pm10_factor': 1.2}),
    'Jawa Timur': (-7.2575, 112.7521, {'pm25_factor': 1.0, 'pm10_factor': 1.1}),
    'Jawa Tengah': (-6.9914, 110.4208, {'pm25_factor': 1.05, 'pm10_factor': 1.15}),
    'DI Yogyakarta': (-7.7956, 110.3695, {'pm25_factor': 0.9, 'pm10_factor': 1.0}),
    'Banten': (-6.1147, 106.1597, {'pm25_factor': 1.2, 'pm10_factor': 1.3}),
    'Sumatera Utara': (3.5952, 98.6728, {'pm25_factor': 0.8, 'pm10_factor': 0.9}),
    'Kalimantan Timur': (1.4437, 116.7758, {'pm25_factor': 0.7, 'pm10_factor': 0.8}),
    'Sulawesi Selatan': (-5.1474, 119.4101, {'pm25_factor': 0.6, 'pm10_factor': 0.7}),
    'Papua': (-2.5489, 140.6493, {'pm25_factor': 0.5, 'pm10_factor': 0.6}),
    'Maluku': (-3.6954, 128.1828, {'pm25_factor': 0.5, 'pm10_factor': 0.6}),
}

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
    'Surabaya': (-7.2575, 112.7521, {'pm25_factor': 1.15, 'pm10_factor': 1.25}),
    'Malang_Kota': (-7.976, 112.631, {'pm25_factor': 1.05, 'pm10_factor': 1.15}),
    'Batu': (-7.597, 112.525, {'pm25_factor': 0.9, 'pm10_factor': 1.0}),
}

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

def generate_parameters_based_on_location(provinsi, kabupaten=None, distrik=None):
    """Generate realistic parameters based on location"""
    # Base values
    base_pm25 = random.uniform(10, 60)
    base_pm10 = base_pm25 * random.uniform(2.5, 4.0)
    base_co = random.uniform(5, 15)
    base_suhu = random.uniform(25, 32)
    base_kelembaban = random.uniform(65, 85)
    
    # Apply location factors
    if provinsi in PROVINSI_DATA:
        prov_factor = PROVINSI_DATA[provinsi][2]
        base_pm25 *= prov_factor.get('pm25_factor', 1.0)
        base_pm10 *= prov_factor.get('pm10_factor', 1.0)
    
    if kabupaten and kabupaten in JAWA_TIMUR_DAERAH:
        kab_factor = JAWA_TIMUR_DAERAH[kabupaten][2]
        base_pm25 *= kab_factor.get('pm25_factor', 1.0)
        base_pm10 *= kab_factor.get('pm10_factor', 1.0)
    
    if distrik and distrik in SURABAYA_DISTRIK:
        distrik_factor = SURABAYA_DISTRIK[distrik][2]
        base_pm25 *= distrik_factor.get('pm25_factor', 1.0)
        base_pm10 *= distrik_factor.get('pm10_factor', 1.0)
    
    # Add some randomness
    pm25 = max(0, base_pm25 + random.uniform(-5, 5))
    pm10 = max(0, base_pm10 + random.uniform(-10, 10))
    co = max(0.1, base_co + random.uniform(-2, 2))
    suhu = max(15, min(40, base_suhu + random.uniform(-2, 2)))
    kelembaban = max(30, min(100, base_kelembaban + random.uniform(-5, 5)))
    
    return {
        'PM2.5': round(pm25, 1),
        'PM10': round(pm10, 1),
        'CO': round(co, 1),
        'Suhu': round(suhu, 1),
        'Kelembaban': round(kelembaban, 1)
    }

# Load model dengan caching
@st.cache_resource
def load_model_indonesia():
    """Load trained model dan preprocessing info"""
    try:
        model = joblib.load('rf_model_indonesia.pkl')
        preprocessor_info = joblib.load('preprocessor_info.pkl')
        label_map = joblib.load('label_map.pkl')
        return model, preprocessor_info, label_map
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

def predict_quality_indonesia(features, model, preprocessor_info):
    """
    Predict kualitas udara berdasarkan features
    """
    try:
        # Load label map
        label_map = joblib.load('label_map.pkl')
        inv_label_map = {v: k for k, v in label_map.items()}
        
        numeric_features = preprocessor_info['numeric_features']
        prov_cols = preprocessor_info['provinsi_columns']
        scaler = preprocessor_info['scaler']

        # Build numeric row vector
        X_num = np.array([[float(features.get(col, 0.0)) for col in numeric_features]])
        X_num_scaled = scaler.transform(X_num)

        # Build prov one-hot
        prov = str(features.get('Provinsi', 'Unknown'))
        prov_df = pd.get_dummies(pd.Series([prov]), prefix='Prov')
        
        # Reindex ke prov_cols
        prov_row = pd.DataFrame(0.0, columns=prov_cols, index=[0])
        for c in prov_cols:
            if c in prov_df.columns:
                prov_row.loc[0, c] = 1.0

        X_pre = np.hstack([X_num_scaled, prov_row.values])
        
        # Predict
        pred_num = model.predict(X_pre)[0]
        probs = model.predict_proba(X_pre)[0]
        maxp = float(np.max(probs))
        label_str = inv_label_map.get(int(pred_num), "Unknown")
        return label_str, maxp
        
    except Exception as e:
        st.error(f"Error dalam prediksi: {e}")
        # Fallback ke perhitungan manual berdasarkan PM2.5
        pm25 = features.get('PM2.5', 0)
        manual_label = _label_from_pm25(pm25)
        return manual_label, 0.8  # Confidence default

# UI Setup
st.set_page_config(page_title="Sistem Kualitas Udara Indonesia", layout="wide")
st.title("🌫️ Sistem Cerdas Deteksi Kualitas Udara Indonesia")
st.markdown("Monitor dan prediksi kualitas udara di berbagai wilayah Indonesia")

# Load model
model, preprocessor_info, label_map = load_model_indonesia()
MODEL_READY = model is not None and preprocessor_info is not None

if MODEL_READY:
    st.sidebar.success("✅ Sistem siap digunakan")
else:
    st.sidebar.warning("⚠️ Sistem belum tersedia, menggunakan analisis manual")

# Sidebar controls
st.sidebar.header("📍 Pilih Lokasi")
provinsi = st.sidebar.selectbox("Provinsi", list(PROVINSI_DATA.keys()), index=2)

kabupaten = None
distrik = None

if provinsi == 'Jawa Timur':
    kabupaten_options = ['Pilih Kabupaten/Kota'] + list(JAWA_TIMUR_DAERAH.keys())
    kabupaten = st.sidebar.selectbox("Kabupaten/Kota", kabupaten_options, index=0)

    if kabupaten == 'Surabaya' and kabupaten != 'Pilih Kabupaten/Kota':
        distrik_options = ['Pilih Kecamatan'] + list(SURABAYA_DISTRIK.keys())
        distrik = st.sidebar.selectbox("Kecamatan", distrik_options, index=0)

# Tombol untuk generate parameter otomatis berdasarkan lokasi
if st.sidebar.button("🎯 Generate Parameter Berdasarkan Lokasi"):
    if provinsi == 'Pilih Provinsi':
        st.sidebar.warning("Pilih provinsi terlebih dahulu!")
    else:
        # Generate parameters berdasarkan lokasi
        generated_params = generate_parameters_based_on_location(provinsi, 
                                                               kabupaten if kabupaten != 'Pilih Kabupaten/Kota' else None,
                                                               distrik if distrik != 'Pilih Kecamatan' else None)
        
        # Simpan ke session state
        synthetic_data = {
            **generated_params,
            'Provinsi': provinsi,
            'Kabupaten': kabupaten if kabupaten != 'Pilih Kabupaten/Kota' else 'Umum',
            'Distrik': distrik if distrik != 'Pilih Kecamatan' else 'Umum'
        }
        
        st.session_state['current_data'] = synthetic_data
        st.session_state['generated_params'] = generated_params
        st.success("✅ Parameter berhasil di-generate berdasarkan lokasi!")

# Juga berikan opsi input manual
st.sidebar.header("📊 Atur Parameter Manual (Opsional)")
pm25 = st.sidebar.slider("PM2.5 (μg/m³)", 0.0, 200.0, 25.0, 0.1)
pm10 = st.sidebar.slider("PM10 (μg/m³)", 0.0, 500.0, 60.0, 1.0)
co = st.sidebar.slider("CO (ppm)", 0.0, 50.0, 8.0, 0.1)
suhu = st.sidebar.slider("Suhu (°C)", 15.0, 40.0, 28.0, 0.1)
kelembaban = st.sidebar.slider("Kelembaban (%)", 30.0, 100.0, 75.0, 0.1)

if st.sidebar.button("🔄 Gunakan Parameter Manual"):
    synthetic_data = {
        'PM2.5': pm25,
        'PM10': pm10,
        'CO': co,
        'Suhu': suhu,
        'Kelembaban': kelembaban,
        'Provinsi': provinsi,
        'Kabupaten': kabupaten if kabupaten != 'Pilih Kabupaten/Kota' else 'Umum',
        'Distrik': distrik if distrik != 'Pilih Kecamatan' else 'Umum'
    }
    
    st.session_state['current_data'] = synthetic_data
    st.success("✅ Data manual berhasil diterapkan!")

# Display results
if 'current_data' in st.session_state:
    data = st.session_state['current_data']
    
    # Header
    lokasi_text = f"{provinsi}"
    if kabupaten and kabupaten != 'Pilih Kabupaten/Kota':
        lokasi_text += f" - {kabupaten}"
    if distrik and distrik != 'Pilih Kecamatan':
        lokasi_text += f" - {distrik}"
    
    st.header(f"📊 Hasil Analisis: {lokasi_text}")
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("PM2.5", f"{data['PM2.5']:.1f} μg/m³")
    with col2:
        st.metric("PM10", f"{data['PM10']:.1f} μg/m³")
    with col3:
        st.metric("CO", f"{data['CO']:.1f} ppm")
    with col4:
        st.metric("Suhu", f"{data['Suhu']:.1f} °C")
    with col5:
        st.metric("Kelembaban", f"{data['Kelembaban']:.1f}%")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 4))
    parameters = ['PM2.5', 'PM10', 'CO', 'Suhu', 'Kelembaban']
    values = [data['PM2.5'], data['PM10'], data['CO'], data['Suhu'], data['Kelembaban']]
    
    # Color based on PM2.5 level
    pm25_level = _label_from_pm25(data['PM2.5'])
    color_map = {
        'Aman': 'green',
        'Waspada': 'yellow', 
        'Tidak Aman bagi Rentan': 'orange',
        'Tidak Aman': 'red'
    }
    main_color = color_map.get(pm25_level, 'gray')
    colors = [main_color] + ['lightgray'] * 4
    
    bars = ax.bar(parameters, values, color=colors, alpha=0.7)
    ax.set_ylabel('Nilai')
    ax.set_title('Parameter Kualitas Udara')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{value:.1f}', 
                ha='center', va='bottom', fontweight='bold')
    
    st.pyplot(fig)
    
    # Map
    st.subheader("🗺️ Peta Lokasi")
    try:
        if provinsi == 'Jawa Timur' and kabupaten == 'Surabaya' and distrik and distrik != 'Pilih Kecamatan':
            lat, lon, _ = SURABAYA_DISTRIK[distrik]
        elif provinsi == 'Jawa Timur' and kabupaten and kabupaten != 'Pilih Kabupaten/Kota':
            lat, lon, _ = JAWA_TIMUR_DAERAH[kabupaten]
        else:
            lat, lon, _ = PROVINSI_DATA[provinsi]
        
        m = folium.Map(location=[lat, lon], zoom_start=10)
        folium.Marker(
            [lat, lon],
            popup=f"{lokasi_text}\nPM2.5: {data['PM2.5']:.1f} μg/m³",
            tooltip="Lokasi Pemantauan"
        ).add_to(m)
        folium_static(m)
    except Exception as e:
        st.error(f"Error menampilkan peta: {e}")
    
    # Prediction and Recommendation
    st.header("💡 Rekomendasi Aktivitas")
    
    # FIX: Pastikan features lengkap sebelum prediksi
    features = {
        'PM2.5': data['PM2.5'],
        'PM10': data['PM10'],
        'CO': data['CO'],
        'Suhu': data['Suhu'],
        'Kelembaban': data['Kelembaban'],
        'Provinsi': data['Provinsi']
    }
    
    try:
        if MODEL_READY:
            label, confidence = predict_quality_indonesia(features, model, preprocessor_info)
        else:
            # Fallback to manual calculation if model not ready
            label = _label_from_pm25(data['PM2.5'])
            confidence = 0.8
            
        # Display result dengan warna yang sesuai
        color_boxes = {
            'Aman': '🟢',
            'Waspada': '🟡', 
            'Tidak Aman bagi Rentan': '🟠',
            'Tidak Aman': '🔴'
        }
        
        emoji = color_boxes.get(label, '⚪')
        
        if label == 'Aman':
            st.success(f"## {emoji} LEVEL: {label}")
            if MODEL_READY:
                st.write(f"*Tingkat kepercayaan: {confidence:.1%}*")
            st.info("""
            ✅ **SEMUA AKTIVITAS AMAN**
            - Olahraga luar ruangan bebas dilakukan
            - Bersepeda, jogging, bermain di taman, piknik aman
            - Tidak perlu pembatasan aktivitas
            - Cocok juga untuk anak-anak, lansia, dan penderita penyakit pernapasan.
            """)
        elif label == 'Waspada':
            st.warning(f"## {emoji} LEVEL: {label}")
            if MODEL_READY:
                st.write(f"*Tingkat kepercayaan: {confidence:.1%}*")
            st.info("""
            ⚠️ **HATI-HATI**
            - Aktivitas ringan-sedang masih boleh dilakukan
            - Jalan santai, bersepeda ringan, atau olahraga pagi/sore
            - Kelompok rentan perlu berhati-hati
            - Hindari aktivitas berat berkepanjangan
            """)
        elif label == 'Tidak Aman bagi Rentan':
            st.error(f"## {emoji} LEVEL: {label}")
            if MODEL_READY:
                st.write(f"*Tingkat kepercayaan: {confidence:.1%}*")
            st.info("""
            🚫 **BATASI AKTIVITAS**
            - Kelompok rentan (anak-anak, lansia, dan penderita asma/penyakit paru) sebaiknya di dalam ruangan
            - Bagi yang sehat, aktivitas ringan masih bisa dilakukan
            - Gunakan masker jika harus keluar
            """)
        else:
            st.error(f"## {emoji} LEVEL: {label}")
            if MODEL_READY:
                st.write(f"*Tingkat kepercayaan: {confidence:.1%}*")
            st.info("""
            ❌ **HINDARI AKTIVITAS LUAR**
            - Semua orang sebaiknya di dalam ruangan
            - Gunakan air purifier jika memungkinkan
            - Tunda aktivitas luar yang tidak penting
            - Jika terpaksa, gunakan pelindung pernapasan yang sesuai dan batasi waktu di luar seminimal mungkin
            """)
            
    except Exception as e:
        st.error(f"Error dalam analisis: {e}")
        # Ultimate fallback
        manual_label = _label_from_pm25(data['PM2.5'])
        st.warning(f"## Analisis Manual: {manual_label}")
        st.info(_aktivitas_rekomendasi(manual_label))

else:
    st.info("👈 Pilih lokasi dan klik 'Generate Parameter Berdasarkan Lokasi' untuk memulai")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Klasifikasi Kualitas Udara:**
- 🟢 **Aman**: PM2.5 < 15 μg/m³
- 🟡 **Waspada**: 15-35 μg/m³  
- 🟠 **Tidak Aman bagi Rentan**: 35-55 μg/m³
- 🔴 **Tidak Aman**: >55 μg/m³
""")