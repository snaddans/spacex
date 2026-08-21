import pandas as pd
import numpy as np
import requests
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("1. Extracting data from Wikipedia...")
url = "https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches"

headers = {
    'User-Agent': 'SpaceXLaunchPredictor/1.0 (Data Science Portfolio Project)'
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    html_data = io.StringIO(response.text)
    tables = pd.read_html(html_data)
    flight_tables = [t for t in tables if any(c in str(t.columns) for c in ['Booster landing', 'Launch outcome', 'Outcome'])]
else:
    print(f"Failed to fetch Wikipedia page. Status code: {response.status_code}")
    flight_tables = []

if not flight_tables:
    print("Generating fallback dataset...")
    df = pd.DataFrame({
        'Flight No.': range(1, 101),
        'Rocket': ['Falcon 9'] * 95 + ['Falcon Heavy'] * 5,
        'Payload mass (kg)': np.random.randint(2000, 15000, 100),
        'Orbit': np.random.choice(['LEO', 'GTO', 'ISS', 'PO'], 100),
        'Launch site': np.random.choice(['CCSFS', 'KSC', 'VSFB'], 100),
        'Booster landing': np.random.choice(['Success', 'Failure', 'No attempt'], 100)
    })
else:
    df = pd.concat(flight_tables, ignore_index=True)
    df = df.loc[:, ~df.columns.duplicated()]
    print("✅ Wikipedia data successfully extracted!")

print("2. Cleaning the Dataset...")
df.columns = [col.strip() for col in df.columns]

payload_cols = [c for c in df.columns if 'Payload mass' in c]
if payload_cols:
    df['Payload mass (kg)'] = df[payload_cols[0]]

orbit_cols = [c for c in df.columns if 'Orbit' in c]
if orbit_cols:
    df['Orbit'] = df[orbit_cols[0]]

# ==========================================
# FILTERING SPECIFICALLY FOR FALCON 9
# ==========================================
if 'Rocket' in df.columns:
    df = df[df['Rocket'].str.contains('Falcon 9', na=False)]
    print("✅ Filtered dataset to only include Falcon 9 launches.")

# Clean Payload Mass
if 'Payload mass (kg)' in df.columns:
    df['Payload mass (kg)'] = df['Payload mass (kg)'].astype(str).str.replace(',', '', regex=False)
    df['Payload mass (kg)'] = df['Payload mass (kg)'].str.extract(r'(\d+)').astype(float)
    df['Payload mass (kg)'] = df['Payload mass (kg)'].fillna(df['Payload mass (kg)'].mean())

# Clean Launch Site
if 'Launch site' in df.columns:
    def clean_site(site):
        site = str(site)
        if 'CCSFS' in site or 'Cape' in site or 'SLC-40' in site: return 'CCSFS'
        if 'KSC' in site or 'Kennedy' in site or 'LC-39A' in site: return 'KSC'
        if 'VSFB' in site or 'Vandenberg' in site or 'SLC-4E' in site: return 'VSFB'
        return 'Other'
    df['Launch site'] = df['Launch site'].apply(clean_site)

# STRICT ORBIT FILTERING
if 'Orbit' in df.columns:
    def clean_orbit(orbit):
        orbit = str(orbit).upper()
        if 'LEO' in orbit or 'VLEO' in orbit: return 'LEO'
        elif 'GTO' in orbit or 'GEO' in orbit: return 'GTO'
        elif 'SSO' in orbit or 'SUN' in orbit: return 'SSO'
        elif 'PO' in orbit or 'POLAR' in orbit: return 'PO'
        elif 'ISS' in orbit: return 'ISS'
        elif 'MEO' in orbit: return 'MEO'
        elif 'HEO' in orbit: return 'HEO'
        elif 'TLI' in orbit: return 'TLI'
        else: return 'Other' 
    
    df['Orbit'] = df['Orbit'].apply(clean_orbit)

print("3. Preparing Features and Target...")
target_col = None
for candidate in ['Booster landing', 'Launch outcome', 'Outcome']:
    if candidate in df.columns:
        target_col = candidate
        break

if not target_col:
    target_col = df.columns[-1]

df['Landing_Success'] = df[target_col].astype(str).str.contains('Success', case=False, na=False).astype(int)

# ==========================================
# EXPORT EXPLICITLY NAMED FALCON 9 DATA TO CSV
# ==========================================
csv_save_path = r"C:\Users\shash_8gna92j\OneDrive\Desktop\project\resume pro\spacex\falcon_9_filtered_data.csv"
df.to_csv(csv_save_path, index=False)
print(f"✅ Falcon 9 filtered dataset successfully saved to: {csv_save_path}")
# ==========================================

features = ['Payload mass (kg)', 'Orbit', 'Launch site']
features = [f for f in features if f in df.columns]

X = df[features].copy()
y = df['Landing_Success']

# Encode Categorical Variables
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = X[col].fillna("Unknown") 
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("4. Training Model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print(f"\n✅ Model Accuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")

print("5. Overwriting old data...")
model_save_path = r"C:\Users\shash_8gna92j\OneDrive\Desktop\project\resume pro\spacex\spacex_model.joblib"

if os.path.exists(model_save_path):
    os.remove(model_save_path)
    print("🗑️ Deleted old corrupted model.")

model_package = {
    'model': rf_model,
    'encoders': label_encoders,
    'features': features,
    'df': df 
}

joblib.dump(model_package, model_save_path)
print(f"✅ New clean model successfully saved to: {model_save_path}")