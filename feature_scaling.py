import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load dataset bersih tanpa outlier
train_clean = pd.read_csv('sample_data/house_pricing_clean.csv')

# Pilih fitur numerik yang akan di-scaling
features = ['GrLivArea', 'SalePrice']

# 1. Scaling dengan StandardScaler
scaler_standard = StandardScaler()
train_standard = pd.DataFrame(scaler_standard.fit_transform(train_clean[features]), columns=features)

# 2. Scaling dengan MinMaxScaler
scaler_minmax = MinMaxScaler()
train_minmax = pd.DataFrame(scaler_minmax.fit_transform(train_clean[features]), columns=features)

# 3. Visualisasi Histogram Sebelum Scaling
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(train_clean['GrLivArea'], kde=True, color='blue')
plt.title('Distribusi GrLivArea Sebelum Scaling')

plt.subplot(1, 2, 2)
sns.histplot(train_clean['SalePrice'], kde=True, color='orange')
plt.title('Distribusi SalePrice Sebelum Scaling')

plt.tight_layout()
plt.savefig('histogram_sebelum_scaling.png', dpi=300)
plt.show()

# 4. Visualisasi Histogram Setelah StandardScaler
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(train_standard['GrLivArea'], kde=True, color='blue')
plt.title('Distribusi GrLivArea dengan StandardScaler')

plt.subplot(1, 2, 2)
sns.histplot(train_standard['SalePrice'], kde=True, color='orange')
plt.title('Distribusi SalePrice dengan StandardScaler')

plt.tight_layout()
plt.savefig('histogram_standard_scaler.png', dpi=300)
plt.show()

# 5. Visualisasi Histogram Setelah MinMaxScaler
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(train_minmax['GrLivArea'], kde=True, color='blue')
plt.title('Distribusi GrLivArea dengan MinMaxScaler')

plt.subplot(1, 2, 2)
sns.histplot(train_minmax['SalePrice'], kde=True, color='orange')
plt.title('Distribusi SalePrice dengan MinMaxScaler')

plt.tight_layout()
plt.savefig('histogram_minmax_scaler.png', dpi=300)
plt.show()

# Download hasil gambar di Google Colab
from google.colab import files
files.download('histogram_sebelum_scaling.png')
files.download('histogram_standard_scaler.png')
files.download('histogram_minmax_scaler.png')
