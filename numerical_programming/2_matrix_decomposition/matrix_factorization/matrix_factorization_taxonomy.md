# Matrix Factorization: Таксономія та Класифікація

## Зміст

1. [Вступ: Що таке Matrix Factorization](#вступ-що-таке-matrix-factorization)
2. [Загальна Таксономія](#загальна-таксономія)
3. [Exact Decomposition (Точні Методи)](#exact-decomposition-точні-методи)
4. [Approximate Decomposition (Апроксимаційні Методи)](#approximate-decomposition-апроксимаційні-методи)
5. [Statistical Methods (Статистичні Методи)](#statistical-methods-статистичні-методи)
6. [Machine Learning Methods (ML Методи)](#machine-learning-methods-ml-методи)
7. [Порівняльні Таблиці](#порівняльні-таблиці)
8. [Коли Що Використовувати](#коли-що-використовувати)
9. [Практичні Приклади](#практичні-приклади)

---

## Вступ: Що таке Matrix Factorization

**Matrix Factorization** — це загальний термін для методів, які розкладають матрицю на добуток простіших матриць.

### Загальна Ідея

```
A = B × C
```

Де:
- **A** — оригінальна матриця (m × n)
- **B** — перша факторизована матриця (m × k)
- **C** — друга факторизована матриця (k × n)
- **k** — розмірність (зазвичай k << min(m,n))

### Навіщо Факторизувати Матриці?

**1. Компресія даних**
```
Original: 1000 × 1000 = 1,000,000 елементів
Factorized: (1000 × 10) + (10 × 1000) = 20,000 елементів
Compression ratio: 50x
```

**2. Виявлення патернів**
- Латентні структури в даних
- Приховані зв'язки
- Кластери та групи

**3. Зменшення noise**
- Фільтрація шуму
- Smoothing даних
- Деnoising

**4. Передбачення**
- Рекомендаційні системи
- Заповнення пропусків
- Прогнозування

---

## Загальна Таксономія

```
Matrix Factorization / Decomposition
│
├── 1. EXACT METHODS (Точні математичні розклади)
│   │   - Лінійна алгебра
│   │   - Завжди точні
│   │   - Аналітичні рішення
│   │
│   ├── 1.1 SVD (Singular Value Decomposition)
│   ├── 1.2 QR Decomposition
│   ├── 1.3 LU Decomposition
│   ├── 1.4 Cholesky Decomposition
│   ├── 1.5 Eigendecomposition
│   └── 1.6 Schur Decomposition
│
├── 2. APPROXIMATE METHODS (Апроксимаційні методи)
│   │   - Наближені рішення
│   │   - Швидші для великих матриць
│   │   - Trade-off: швидкість vs точність
│   │
│   ├── 2.1 Truncated SVD
│   ├── 2.2 Randomized SVD
│   ├── 2.3 Incremental SVD
│   └── 2.4 Sparse SVD
│
├── 3. STATISTICAL METHODS (Статистичні методи)
│   │   - Фокус на variance/correlation
│   │   - Interpretable components
│   │   - Feature extraction
│   │
│   ├── 3.1 PCA (Principal Component Analysis)
│   ├── 3.2 Factor Analysis (FA)
│   ├── 3.3 ICA (Independent Component Analysis)
│   └── 3.4 CCA (Canonical Correlation Analysis)
│
└── 4. MACHINE LEARNING METHODS (ML для рекомендацій)
    │   - Sparse data
    │   - Iterative optimization
    │   - Prediction focused
    │
    ├── 4.1 Basic Matrix Factorization
    │   ├── Gradient Descent MF
    │   ├── ALS (Alternating Least Squares)
    │   └── SGD (Stochastic Gradient Descent)
    │
    ├── 4.2 Constrained Methods
    │   ├── NMF (Non-negative Matrix Factorization)
    │   └── Binary Matrix Factorization
    │
    ├── 4.3 Probabilistic Methods
    │   ├── PMF (Probabilistic Matrix Factorization)
    │   ├── Bayesian PMF
    │   └── BPMF (Bayesian Probabilistic MF)
    │
    ├── 4.4 Advanced CF Methods
    │   ├── SVD++ (with implicit feedback)
    │   ├── timeSVD++ (temporal dynamics)
    │   ├── SVDFeature (with side information)
    │   └── Factorization Machines
    │
    ├── 4.5 Deep Learning Methods
    │   ├── Neural Collaborative Filtering (NCF)
    │   ├── Autoencoders
    │   ├── Variational Autoencoders (VAE)
    │   └── Deep Matrix Factorization
    │
    └── 4.6 Tensor Factorization
        ├── Tucker Decomposition
        ├── CP Decomposition (CANDECOMP/PARAFAC)
        └── Tensor Train
```

---

## Exact Decomposition (Точні Методи)

### 1.1 SVD (Singular Value Decomposition)

**Формула:**
```
A = U × Σ × V^T
```

**Характеристики:**
- Працює для **будь-якої** матриці (прямокутної)
- Завжди існує
- U і V — ортогональні матриці
- Σ — діагональна матриця з singular values

**Математика:**

```python
import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])  # 4×3 matrix

# Повний SVD
U, s, Vt = np.linalg.svd(A, full_matrices=True)

print(f"A shape: {A.shape}")           # (4, 3)
print(f"U shape: {U.shape}")           # (4, 4) — квадратна
print(f"s shape: {s.shape}")           # (3,) — діагональ
print(f"Vt shape: {Vt.shape}")         # (3, 3) — квадратна

# Реконструкція
Sigma = np.zeros((4, 3))
Sigma[:3, :3] = np.diag(s)
A_reconstructed = U @ Sigma @ Vt

print(f"Reconstruction error: {np.allclose(A, A_reconstructed)}")  # True
```

**Властивості:**

1. **Singular Values** (в Σ) відсортовані за спаданням:
   ```
   σ₁ ≥ σ₂ ≥ σ₃ ≥ ... ≥ 0
   ```

2. **Energy Compaction**: Перші кілька singular values містять більшість інформації

3. **Rank**: Кількість ненульових singular values = rank матриці

4. **Moore-Penrose Pseudoinverse**: 
   ```
   A⁺ = V × Σ⁺ × U^T
   ```

**Застосування:**
- Image compression
- Data denoising
- Solving linear systems
- Low-rank approximation
- Recommender systems (базова версія)

**Приклад: Image Compression**

```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Завантажити зображення
img = Image.open('photo.jpg').convert('L')  # grayscale
A = np.array(img, dtype=float)

# SVD
U, s, Vt = np.linalg.svd(A, full_matrices=False)

# Compression з різними k
for k in [5, 10, 50, 100]:
    # Truncated reconstruction
    A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    
    # Compression ratio
    original_size = A.shape[0] * A.shape[1]
    compressed_size = k * (A.shape[0] + A.shape[1] + 1)
    ratio = original_size / compressed_size
    
    print(f"k={k}: compression ratio = {ratio:.1f}x")
    
    # Відобразити
    plt.subplot(2, 2, k//25)
    plt.imshow(A_k, cmap='gray')
    plt.title(f'k={k} ({ratio:.1f}x)')
    plt.axis('off')

plt.tight_layout()
plt.show()
```

**Обчислювальна Складність:**
- Час: O(min(m²n, mn²))
- Пам'ять: O(mn)

---

### 1.2 QR Decomposition

**Формула:**
```
A = Q × R
```

**Характеристики:**
- **Q** — ортогональна матриця (m × m)
- **R** — верхня трикутна матриця (m × n)
- Використовується для solving linear systems

**Приклад:**

```python
import numpy as np

A = np.array([
    [12, -51, 4],
    [6, 167, -68],
    [-4, 24, -41]
], dtype=float)

# QR decomposition
Q, R = np.linalg.qr(A)

print("Q (orthogonal):")
print(Q)
print("\nR (upper triangular):")
print(R)

# Перевірка
print("\nQ @ R = A:", np.allclose(Q @ R, A))
print("Q^T @ Q = I:", np.allclose(Q.T @ Q, np.eye(3)))
```

**Застосування:**
- Solving Ax = b
- Least squares regression
- Eigenvalue algorithms
- Gram-Schmidt orthogonalization

---

### 1.3 LU Decomposition

**Формула:**
```
A = L × U
```

**Характеристики:**
- **L** — нижня трикутна матриця (lower)
- **U** — верхня трикутна матриця (upper)
- Швидше для розв'язання систем рівнянь

**Приклад:**

```python
from scipy.linalg import lu

A = np.array([
    [2, 1, 1],
    [4, -6, 0],
    [-2, 7, 2]
], dtype=float)

# LU decomposition
P, L, U = lu(A)

print("P (permutation):")
print(P)
print("\nL (lower triangular):")
print(L)
print("\nU (upper triangular):")
print(U)

# Перевірка
print("\nP @ L @ U = A:", np.allclose(P @ L @ U, A))
```

**Застосування:**
- Solving Ax = b (багато right-hand sides)
- Matrix inversion
- Determinant calculation

---

### 1.4 Cholesky Decomposition

**Формула:**
```
A = L × L^T
```

**Характеристики:**
- Працює **тільки** для symmetric positive-definite matrices
- **L** — нижня трикутна матриця
- Швидше ніж LU (в 2 рази)

**Приклад:**

```python
from scipy.linalg import cholesky

# Symmetric positive-definite matrix
A = np.array([
    [4, 12, -16],
    [12, 37, -43],
    [-16, -43, 98]
], dtype=float)

# Cholesky decomposition
L = cholesky(A, lower=True)

print("L:")
print(L)

# Перевірка
print("\nL @ L^T = A:", np.allclose(L @ L.T, A))
```

**Застосування:**
- Gaussian processes
- Monte Carlo simulations
- Linear regression (normal equations)
- Covariance matrices

---

### 1.5 Eigendecomposition

**Формула:**
```
A = Q × Λ × Q^(-1)
```

**Характеристики:**
- Працює для **квадратних** матриць
- **Q** — матриця eigenvectors
- **Λ** — діагональна матриця eigenvalues
- Якщо A symmetric → Q orthogonal

**Приклад:**

```python
A = np.array([
    [2, 1],
    [1, 2]
], dtype=float)

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)

# Реконструкція
Lambda = np.diag(eigenvalues)
A_reconstructed = eigenvectors @ Lambda @ np.linalg.inv(eigenvectors)

print("\nReconstruction:", np.allclose(A, A_reconstructed))
```

**Застосування:**
- PCA (Principal Component Analysis)
- Stability analysis
- Markov chains
- Quantum mechanics
- Vibrational modes

---

### Порівняння Exact Methods

| Метод | Матриці | Унікальність | Складність | Основне Використання |
|-------|---------|--------------|------------|----------------------|
| **SVD** | Будь-які (m×n) | Так | O(mn²) | Compression, recommender systems |
| **QR** | Будь-які (m×n) | Ні | O(mn²) | Linear systems, least squares |
| **LU** | Квадратні (n×n) | Ні | O(n³) | Linear systems (multiple RHS) |
| **Cholesky** | Symmetric PD (n×n) | Так | O(n³/2) | Covariance, Gaussian processes |
| **Eigen** | Квадратні (n×n) | Ні | O(n³) | PCA, stability, spectral analysis |

---

## Approximate Decomposition (Апроксимаційні Методи)

### 2.1 Truncated SVD

**Ідея:** Зберегти тільки топ-k singular values/vectors.

**Формула:**
```
A ≈ U_k × Σ_k × V_k^T
```

Де k << min(m, n)

**Приклад:**

```python
from sklearn.decomposition import TruncatedSVD

# Велика sparse матриця
from scipy.sparse import random
A_sparse = random(1000, 500, density=0.01, format='csr')

# Truncated SVD
svd = TruncatedSVD(n_components=50, random_state=42)
U_k = svd.fit_transform(A_sparse)
Vt_k = svd.components_

print(f"Original: {A_sparse.shape}")
print(f"U_k: {U_k.shape}")
print(f"Vt_k: {Vt_k.shape}")
print(f"Explained variance: {svd.explained_variance_ratio_.sum():.2%}")

# Reconstruction
A_approx = U_k @ Vt_k
```

**Переваги:**
- Набагато швидше для великих матриць
- Менше пам'яті
- Працює з sparse matrices

**Недоліки:**
- Втрата інформації
- Апроксимація

**Застосування:**
- Text mining (LSA - Latent Semantic Analysis)
- Large-scale recommender systems
- Information retrieval

---

### 2.2 Randomized SVD

**Ідея:** Використовувати randomization для прискорення.

**Алгоритм:**
```
1. Генерувати випадкову матрицю Ω (n × k)
2. Y = A × Ω
3. Orthogonalize Y → Q
4. B = Q^T × A
5. SVD на малій матриці B
```

**Приклад:**

```python
from sklearn.utils.extmath import randomized_svd

# Велика матриця
A = np.random.randn(10000, 5000)

# Randomized SVD (швидше!)
U, s, Vt = randomized_svd(
    A, 
    n_components=100,
    n_iter=5,
    random_state=42
)

print(f"U: {U.shape}")
print(f"s: {s.shape}")
print(f"Vt: {Vt.shape}")
```

**Переваги:**
- Дуже швидко для великих матриць
- O(mnk) замість O(mn²)
- Добра апроксимація

**Застосування:**
- Big data applications
- Real-time systems
- Large-scale machine learning

---

### 2.3 Incremental SVD

**Ідея:** Оновлювати SVD коли додаються нові дані.

**Приклад:**

```python
from sklearn.decomposition import IncrementalPCA

# Incremental PCA (uses SVD)
ipca = IncrementalPCA(n_components=10, batch_size=100)

# Streaming data
for batch in data_batches:
    ipca.partial_fit(batch)

# Transform
X_transformed = ipca.transform(X_test)
```

**Переваги:**
- Не потрібно зберігати всі дані в пам'яті
- Підтримує streaming data
- Можна оновлювати модель

**Застосування:**
- Online learning
- Streaming data
- Memory-constrained environments

---

## Statistical Methods (Статистичні Методи)

### 3.1 PCA (Principal Component Analysis)

**Що це:** Статистична техніка для зменшення dimensionality, яка зберігає максимум variance.

**Математика:**

```
1. Центрування: X_centered = X - mean(X)
2. Covariance matrix: Cov = (1/n) × X_centered^T × X_centered
3. Eigendecomposition: Cov = Q × Λ × Q^T
4. Principal components = eigenvectors з найбільшими eigenvalues
```

**Під капотом PCA використовує SVD:**

```python
import numpy as np
from sklearn.decomposition import PCA

# Дані
X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2.0, 1.6],
    [1.0, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]
])

# PCA через sklearn
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Principal Components (directions):")
print(pca.components_)
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)
print("\nTransformed Data:")
print(X_pca)

# Візуалізація
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Original data
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.8)
plt.arrow(pca.mean_[0], pca.mean_[1], 
          pca.components_[0, 0], pca.components_[0, 1],
          head_width=0.1, head_length=0.1, fc='r', ec='r')
plt.arrow(pca.mean_[0], pca.mean_[1], 
          pca.components_[1, 0], pca.components_[1, 1],
          head_width=0.1, head_length=0.1, fc='b', ec='b')
plt.title('Original Data with Principal Components')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, alpha=0.3)

# Transformed data
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.8)
plt.title('Transformed Data (PCA space)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Ручна Імплементація:**

```python
def manual_pca(X, n_components):
    """
    PCA з нуля використовуючи SVD
    """
    # 1. Центрування
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # 2. SVD
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # 3. Principal components = Vt (eigenvectors)
    components = Vt[:n_components]
    
    # 4. Explained variance
    explained_variance = (s ** 2) / (X.shape[0] - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()
    
    # 5. Transform
    X_transformed = X_centered @ components.T
    
    return X_transformed, components, explained_variance_ratio[:n_components]

# Тестування
X_manual, components, variance_ratio = manual_pca(X, n_components=2)
print("Manual PCA matches sklearn:", np.allclose(np.abs(X_pca), np.abs(X_manual)))
```

**Застосування:**

1. **Dimensionality Reduction**
```python
# Зменшити 1000 features → 50 features
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X_high_dim)
```

2. **Visualization**
```python
# MNIST digits: 784 dimensions → 2D
from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits.data, digits.target

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10')
plt.colorbar()
plt.title('MNIST Digits in 2D (PCA)')
```

3. **Noise Filtering**
```python
# Зберегти тільки компоненти з високою variance
pca = PCA(n_components=0.95)  # 95% variance
X_denoised = pca.fit_transform(X_noisy)
X_reconstructed = pca.inverse_transform(X_denoised)
```

4. **Feature Engineering**
```python
# Додати principal components як нові features
pca = PCA(n_components=10)
pca_features = pca.fit_transform(X)
X_augmented = np.hstack([X, pca_features])
```

**PCA vs SVD:**

```python
# Вони дають той самий результат!

# Метод 1: PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Метод 2: Manual SVD
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
X_svd = U[:, :2] * s[:2]

# Однаковий результат (можливо з іншим знаком)
print(np.allclose(np.abs(X_pca), np.abs(X_svd)))  # True
```

---

### 3.2 Factor Analysis (FA)

**Відмінність від PCA:**

| Аспект | PCA | Factor Analysis |
|--------|-----|-----------------|
| Модель | X = PC × weights | X = Factors × loadings + noise |
| Variance | Total variance | Shared variance (без noise) |
| Мета | Dimensionality reduction | Виявити латентні факторі |
| Noise | Не моделюється | Явно моделюється |

**Модель:**
```
X = Λ × F + ε

X — observed variables
Λ — factor loadings
F — latent factors
ε — unique variance (noise)
```

**Приклад:**

```python
from sklearn.decomposition import FactorAnalysis

# Дані з шумом
X_noisy = X + np.random.normal(0, 0.1, X.shape)

# Factor Analysis
fa = FactorAnalysis(n_components=2, random_state=42)
X_factors = fa.fit_transform(X_noisy)

print("Factor Loadings:")
print(fa.components_)
print("\nNoise Variance:")
print(fa.noise_variance_)

# Порівняння з PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_noisy)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_factors[:, 0], X_factors[:, 1])
plt.title('Factor Analysis')
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title('PCA')
plt.xlabel('PC 1')
plt.ylabel('PC 2')

plt.tight_layout()
plt.show()
```

**Застосування:**
- Психометрія (IQ тести)
- Соціологія (latent traits)
- Фінанси (risk factors)
- Bioinformatics (gene expression)

---

### 3.3 ICA (Independent Component Analysis)

**Відмінність від PCA:**

| Аспект | PCA | ICA |
|--------|-----|-----|
| Критерій | Максимальна variance | Максимальна independence |
| Orthogonality | Так | Ні |
| Distribution | Gaussian OK | Non-Gaussian краще |
| Порядок | Важливий (variance) | Невизначений |

**Приклад: Cocktail Party Problem**

```python
from sklearn.decomposition import FastICA
import numpy as np

# Генерація сигналів
time = np.linspace(0, 8, 4000)
s1 = np.sin(2 * time)  # Signal 1
s2 = np.sign(np.sin(3 * time))  # Signal 2
s3 = np.random.laplace(size=len(time))  # Signal 3

S = np.c_[s1, s2, s3]  # Original sources
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

# Змішування (mixing)
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])
X = S @ A.T  # Mixed signals

# ICA для розділення
ica = FastICA(n_components=3, random_state=42)
S_estimated = ica.fit_transform(X)

# Візуалізація
fig, axes = plt.subplots(3, 3, figsize=(15, 10))

for i in range(3):
    axes[0, i].plot(S[:, i])
    axes[0, i].set_title(f'Original Signal {i+1}')
    
    axes[1, i].plot(X[:, i])
    axes[1, i].set_title(f'Mixed Signal {i+1}')
    
    axes[2, i].plot(S_estimated[:, i])
    axes[2, i].set_title(f'Recovered Signal {i+1}')

plt.tight_layout()
plt.show()
```

**Застосування:**
- Signal separation (EEG, audio)
- Image processing
- Financial time series
- fMRI analysis

---

## Machine Learning Methods (ML Методи)

### 4.1 Basic Matrix Factorization для Collaborative Filtering

**Проблема:** У нас є sparse матриця рейтингів з багатьма пропусками.

```
     Movie1  Movie2  Movie3  Movie4  Movie5
UserA   5      ?       1      ?       4
UserB   ?      4       ?      2       ?
UserC   3      ?       5      ?       1
UserD   ?      5       ?      3       ?
```

**Мета:** Передбачити пропуски (?) для рекомендацій.

**Класичний SVD НЕ працює** бо:
- SVD потребує повну матрицю (без пропусків)
- Заповнення пропусків створює bias

**Рішення: Gradient Descent Matrix Factorization**

#### Gradient Descent MF

**Модель:**
```
r̂_ui = q_i^T × p_u

r̂_ui — predicted rating
q_i — item latent factors
p_u — user latent factors
```

**Loss Function:**
```
L = Σ(r_ui - q_i^T × p_u)² + λ(||q_i||² + ||p_u||²)
     known ratings         regularization
```

**Алгоритм:**

```python
import numpy as np

class MatrixFactorizationGD:
    """
    Matrix Factorization using Gradient Descent
    """
    
    def __init__(self, n_factors=10, learning_rate=0.01, 
                 regularization=0.02, n_epochs=100):
        self.k = n_factors
        self.lr = learning_rate
        self.reg = regularization
        self.n_epochs = n_epochs
        
    def fit(self, R, verbose=True):
        """
        R: ratings matrix (users × items)
           0 = missing rating
        """
        n_users, n_items = R.shape
        
        # Ініціалізація
        np.random.seed(42)
        self.P = np.random.normal(0, 0.1, (n_users, self.k))
        self.Q = np.random.normal(0, 0.1, (n_items, self.k))
        
        # Bias terms
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = np.mean(R[R > 0])
        
        # Training
        self.train_loss = []
        
        for epoch in range(self.n_epochs):
            # Shuffle
            indices = np.argwhere(R > 0)
            np.random.shuffle(indices)
            
            epoch_loss = 0
            
            for u, i in indices:
                # Prediction
                pred = (self.global_bias + 
                       self.user_bias[u] + 
                       self.item_bias[i] + 
                       np.dot(self.P[u], self.Q[i]))
                
                # Error
                error = R[u, i] - pred
                epoch_loss += error ** 2
                
                # Gradient descent
                self.user_bias[u] += self.lr * (error - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (error - self.reg * self.item_bias[i])
                
                P_u_old = self.P[u].copy()
                self.P[u] += self.lr * (error * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (error * P_u_old - self.reg * self.Q[i])
            
            # Regularization term
            epoch_loss += self.reg * (
                np.sum(self.P ** 2) + 
                np.sum(self.Q ** 2) +
                np.sum(self.user_bias ** 2) +
                np.sum(self.item_bias ** 2)
            )
            
            self.train_loss.append(epoch_loss)
            
            if verbose and epoch % 10 == 0:
                rmse = np.sqrt(epoch_loss / len(indices))
                print(f"Epoch {epoch}: RMSE = {rmse:.4f}")
        
        return self
    
    def predict(self, u, i):
        """Predict rating for user u and item i"""
        pred = (self.global_bias + 
               self.user_bias[u] + 
               self.item_bias[i] + 
               np.dot(self.P[u], self.Q[i]))
        return pred
    
    def recommend(self, user_id, R, n=5):
        """Recommend top-n items for user"""
        predictions = []
        for i in range(R.shape[1]):
            if R[user_id, i] == 0:  # Not rated
                pred = self.predict(user_id, i)
                predictions.append((i, pred))
        
        # Sort by prediction
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]

# Використання
R = np.array([
    [5, 3, 0, 1, 0],
    [4, 0, 0, 1, 3],
    [1, 1, 0, 5, 0],
    [1, 0, 0, 4, 0],
    [0, 1, 5, 4, 0]
])

model = MatrixFactorizationGD(n_factors=2, n_epochs=100)
model.fit(R)

# Рекомендації для User 0
recommendations = model.recommend(0, R, n=3)
print("\nTop-3 recommendations for User 0:")
for item_id, score in recommendations:
    print(f"  Item {item_id}: {score:.2f}")
```

**Візуалізація Навчання:**

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(model.train_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.grid(True, alpha=0.3)
plt.show()
```

#### ALS (Alternating Least Squares)

**Ідея:** Замість одночасного оновлення P і Q, фіксуємо один і оптимізуємо інший.

**Алгоритм:**
```
1. Ініціалізація P і Q випадково
2. Repeat until convergence:
   a. Fix Q, optimize P
   b. Fix P, optimize Q
```

**Переваги:**
- Паралелізується легко
- Швидше для великих матриць
- Використовується в Spark MLlib

**Імплементація:**

```python
class MatrixFactorizationALS:
    """
    Matrix Factorization using Alternating Least Squares
    """
    
    def __init__(self, n_factors=10, regularization=0.1, n_iterations=10):
        self.k = n_factors
        self.reg = regularization
        self.n_iter = n_iterations
    
    def fit(self, R):
        """
        R: ratings matrix (users × items)
           0 = missing rating
        """
        n_users, n_items = R.shape
        
        # Ініціалізація
        np.random.seed(42)
        self.P = np.random.normal(0, 0.1, (n_users, self.k))
        self.Q = np.random.normal(0, 0.1, (n_items, self.k))
        
        # Masks
        user_rated = (R > 0)
        
        for iteration in range(self.n_iter):
            # Fix Q, update P
            for u in range(n_users):
                # Items rated by user u
                items = np.where(user_rated[u])[0]
                if len(items) == 0:
                    continue
                
                Q_u = self.Q[items]
                R_u = R[u, items]
                
                # Closed-form solution
                # P_u = (Q_u^T Q_u + λI)^(-1) Q_u^T R_u
                A = Q_u.T @ Q_u + self.reg * np.eye(self.k)
                b = Q_u.T @ R_u
                self.P[u] = np.linalg.solve(A, b)
            
            # Fix P, update Q
            for i in range(n_items):
                # Users who rated item i
                users = np.where(user_rated[:, i])[0]
                if len(users) == 0:
                    continue
                
                P_i = self.P[users]
                R_i = R[users, i]
                
                # Closed-form solution
                A = P_i.T @ P_i + self.reg * np.eye(self.k)
                b = P_i.T @ R_i
                self.Q[i] = np.linalg.solve(A, b)
            
            # Calculate RMSE
            predictions = self.P @ self.Q.T
            mask = R > 0
            rmse = np.sqrt(np.mean((R[mask] - predictions[mask]) ** 2))
            print(f"Iteration {iteration + 1}: RMSE = {rmse:.4f}")
        
        return self
    
    def predict(self, u, i):
        return np.dot(self.P[u], self.Q[i])

# Використання
model_als = MatrixFactorizationALS(n_factors=2, n_iterations=20)
model_als.fit(R)
```

**ALS vs Gradient Descent:**

| Аспект | Gradient Descent | ALS |
|--------|------------------|-----|
| **Швидкість** | Повільніше | Швидше |
| **Паралелізація** | Важко | Легко |
| **Convergence** | Потребує tuning | Більш стабільна |
| **Implicit Feedback** | Складніше | Простіше |
| **Використання** | Small-medium data | Large-scale |

---

### 4.2 NMF (Non-negative Matrix Factorization)

**Обмеження:** Всі елементи P, Q ≥ 0

**Чому це корисно:**

1. **Interpretability**: Додатні значення легше інтерпретувати
2. **Parts-based representation**: Розкладає на "частини"
3. **Sparsity**: Часто дає sparse factors

**Приклад: Image Decomposition**

```python
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

# Завантажити dataset faces
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X = faces.data  # 400 faces, 64×64 pixels

# NMF
n_components = 16
nmf = NMF(n_components=n_components, init='random', random_state=42, max_iter=200)
W = nmf.fit_transform(X)  # Encodings (400 × 16)
H = nmf.components_        # Components (16 × 4096)

# Візуалізувати компоненти (базисні "обличчя")
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(H[i].reshape(64, 64), cmap='gray')
    ax.axis('off')
    ax.set_title(f'Component {i+1}')
plt.suptitle('NMF Components (Face Parts)')
plt.tight_layout()
plt.show()

# Реконструкція обличчя
face_idx = 0
original = X[face_idx].reshape(64, 64)
reconstructed = (W[face_idx] @ H).reshape(64, 64)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')
axes[1].imshow(reconstructed, cmap='gray')
axes[1].set_title('NMF Reconstruction')
axes[1].axis('off')
plt.show()
```

**Застосування:**
- Image analysis
- Text mining (topic modeling)
- Audio source separation
- Bioinformatics (gene expression)

---

### 4.3 PMF (Probabilistic Matrix Factorization)

**Ідея:** Моделювати невизначеність через ймовірнісну модель.

**Модель:**
```
p(R | P, Q, σ²) = Π N(r_ui | p_u^T q_i, σ²)
p(P | σ_P²) = Π N(p_u | 0, σ_P² I)
p(Q | σ_Q²) = Π N(q_i | 0, σ_Q² I)
```

**Переваги:**
- Моделює uncertainty
- Bayesian approach
- Краще для холодного старту

**Приклад (концептуальний):**

```python
class ProbabilisticMF:
    """
    Probabilistic Matrix Factorization
    """
    
    def __init__(self, n_factors=10, sigma=0.1, sigma_P=0.1, sigma_Q=0.1):
        self.k = n_factors
        self.sigma = sigma          # Rating noise
        self.sigma_P = sigma_P      # User prior
        self.sigma_Q = sigma_Q      # Item prior
    
    def fit(self, R, n_iterations=100):
        n_users, n_items = R.shape
        
        # Ініціалізація
        self.P_mean = np.random.normal(0, 0.1, (n_users, self.k))
        self.Q_mean = np.random.normal(0, 0.1, (n_items, self.k))
        
        # Можна також зберігати variance/covariance
        self.P_var = np.ones((n_users, self.k)) * self.sigma_P ** 2
        self.Q_var = np.ones((n_items, self.k)) * self.sigma_Q ** 2
        
        # MAP estimation (similar to regularized MF)
        # але можна робити full Bayesian inference
        
        return self
```

**Застосування:**
- Recommender systems з uncertainty
- Active learning (де запитувати рейтинг?)
- Exploration vs exploitation

---

### 4.4 Advanced CF Methods

#### SVD++ (Koren, 2008)

**Ідея:** Додати implicit feedback.

**Модель:**
```
r̂_ui = μ + b_u + b_i + q_i^T (p_u + |N(u)|^(-1/2) Σ y_j)
                                          j∈N(u)

N(u) — items rated by user u (implicit feedback)
y_j — implicit item factors
```

**Переваги:**
- Використовує implicit signals
- Кращі результати
- Переможець Netflix Prize (частина рішення)

#### timeSVD++ (Koren, 2009)

**Ідея:** Додати temporal dynamics.

**Модель:**
```
r̂_ui(t) = μ + b_u(t) + b_i(t) + q_i^T p_u(t)

Biases та factors змінюються з часом
```

**Застосування:**
- Динамічні рекомендації
- Сезонні ефекти
- Еволюція смаків

---

### 4.5 Deep Learning Methods

#### Neural Collaborative Filtering (NCF)

**Ідея:** Замінити dot product на neural network.

```python
import torch
import torch.nn as nn

class NeuralCF(nn.Module):
    """
    Neural Collaborative Filtering
    """
    
    def __init__(self, n_users, n_items, n_factors=16, layers=[64, 32, 16]):
        super().__init__()
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        
        # MLP layers
        self.fc_layers = nn.ModuleList()
        input_size = n_factors * 2
        
        for layer_size in layers:
            self.fc_layers.append(nn.Linear(input_size, layer_size))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(0.2))
            input_size = layer_size
        
        # Output layer
        self.output = nn.Linear(layers[-1], 1)
    
    def forward(self, user_ids, item_ids):
        # Get embeddings
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        
        # Concatenate
        x = torch.cat([user_vec, item_vec], dim=1)
        
        # MLP
        for layer in self.fc_layers:
            x = layer(x)
        
        # Output
        rating = self.output(x)
        return rating.squeeze()

# Використання
model = NeuralCF(n_users=1000, n_items=500, n_factors=32)

# Training (pseudo-code)
# optimizer = torch.optim.Adam(model.parameters())
# criterion = nn.MSELoss()
# for epoch in range(n_epochs):
#     predictions = model(user_ids, item_ids)
#     loss = criterion(predictions, ratings)
#     loss.backward()
#     optimizer.step()
```

**Переваги:**
- Нелінійні взаємодії
- Може використовувати side information
- State-of-the-art результати

---

### 4.6 Tensor Factorization

**Ідея:** Розширити на 3D+ тензори.

**Приклад: User × Item × Time**

```
User A rated Item 1 with 5 stars on 2024-01-01
→ Tensor[A, 1, Jan-2024] = 5
```

**Tucker Decomposition:**
```
T ≈ G ×₁ U ×₂ I ×₃ C

G — core tensor
U — user factors
I — item factors  
C — context factors
```

**Застосування:**
- Context-aware recommendations
- Time-evolving systems
- Multi-modal data

---

## Порівняльні Таблиці

### Порівняння Всіх Методів

| Метод | Категорія | Sparse OK? | Interpretable? | Складність | Best For |
|-------|-----------|------------|----------------|------------|----------|
| **SVD** | Exact | ❌ | ⭐⭐ | O(mn²) | Image compression, dimensionality reduction |
| **QR** | Exact | ❌ | ⭐ | O(mn²) | Linear systems, least squares |
| **LU** | Exact | ❌ | ⭐ | O(n³) | Solving Ax=b multiple times |
| **Cholesky** | Exact | ❌ | ⭐ | O(n³/2) | Covariance matrices, Gaussian processes |
| **Truncated SVD** | Approximate | ✅ | ⭐⭐ | O(mnk) | Text mining, large sparse matrices |
| **Randomized SVD** | Approximate | ✅ | ⭐⭐ | O(mnk) | Big data, real-time systems |
| **PCA** | Statistical | ❌ | ⭐⭐⭐ | O(mn²) | Feature extraction, visualization |
| **Factor Analysis** | Statistical | ❌ | ⭐⭐⭐ | O(mn²) | Psychology, latent traits |
| **ICA** | Statistical | ❌ | ⭐⭐⭐ | O(mn²) | Signal separation, blind source separation |
| **GD MF** | ML | ✅ | ⭐⭐ | O(nnz·k) | Small-medium recommender systems |
| **ALS** | ML | ✅ | ⭐⭐ | O(nnz·k) | Large-scale collaborative filtering |
| **NMF** | ML | ❌/✅ | ⭐⭐⭐⭐ | O(mnk) | Image/text analysis, parts-based |
| **PMF** | ML | ✅ | ⭐⭐ | O(nnz·k) | Bayesian CF, uncertainty modeling |
| **SVD++** | ML | ✅ | ⭐⭐ | O(nnz·k) | Implicit + explicit feedback |
| **NCF** | Deep Learning | ✅ | ⭐ | O(nnz·k) | State-of-the-art CF |

**Legend:**
- nnz = number of non-zero entries
- k = number of latent factors
- ⭐ = interpretability level

### Вибір Методу: Decision Tree

```
Є sparse матриця (багато пропусків)?
│
├─ НІ → Чи потрібна точна декомпозиція?
│      │
│      ├─ ТАК → Який тип матриці?
│      │      ├─ Будь-яка → SVD
│      │      ├─ Symmetric PD → Cholesky
│      │      └─ Квадратна → LU або Eigen
│      │
│      └─ НІ → Яка мета?
│             ├─ Dimensionality reduction → PCA
│             ├─ Signal separation → ICA
│             ├─ Parts-based → NMF
│             └─ Latent traits → Factor Analysis
│
└─ ТАК → Collaborative Filtering?
       │
       ├─ ТАК → Який розмір даних?
       │      ├─ Small-Medium → Gradient Descent MF
       │      ├─ Large → ALS (Spark)
       │      ├─ Implicit + Explicit → SVD++
       │      └─ State-of-the-art → NCF (Deep Learning)
       │
       └─ НІ → Text/Image analysis?
              ├─ Text → Truncated SVD (LSA)
              ├─ Image parts → NMF
              └─ Large sparse → Randomized SVD
```

---

## Коли Що Використовувати

### Use Case 1: Image Compression

**Завдання:** Стиснути зображення зі збереженням якості.

**Рішення:** SVD або Truncated SVD

```python
from PIL import Image
import numpy as np

# Load image
img = Image.open('photo.jpg').convert('L')
A = np.array(img, dtype=float)

# SVD
U, s, Vt = np.linalg.svd(A, full_matrices=False)

# Compress
for k in [10, 50, 100]:
    A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    
    # Save
    Image.fromarray(A_k.astype('uint8')).save(f'compressed_k{k}.jpg')
```

---

### Use Case 2: Netflix-Style Recommendations

**Завдання:** Рекомендувати фільми користувачам.

**Дані:** Sparse матриця рейтингів (5% заповнена).

**Рішення:** ALS або Gradient Descent MF

```python
# Якщо малий датасет (<100K ratings)
model = MatrixFactorizationGD(n_factors=20, n_epochs=100)
model.fit(ratings_matrix)

# Якщо великий датасет (>1M ratings)
# Використовувати Spark MLlib ALS
from pyspark.ml.recommendation import ALS

als = ALS(
    rank=20,
    maxIter=10,
    regParam=0.01,
    userCol="user_id",
    itemCol="movie_id",
    ratingCol="rating"
)
model = als.fit(ratings_df)
```

---

### Use Case 3: Text Mining (Topic Modeling)

**Завдання:** Виявити topics в документах.

**Дані:** Document-term matrix (дуже sparse).

**Рішення:** Truncated SVD (LSA) або NMF

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF

# Document-term matrix
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(documents)

# Method 1: LSA (Truncated SVD)
lsa = TruncatedSVD(n_components=10)
doc_topics_lsa = lsa.fit_transform(X)

# Method 2: NMF (more interpretable)
nmf = NMF(n_components=10, random_state=42)
doc_topics_nmf = nmf.fit_transform(X)

# Print topics
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(nmf.components_):
    top_words = [feature_names[i] for i in topic.argsort()[-10:]]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")
```

---

### Use Case 4: Dimensionality Reduction для ML

**Завдання:** Зменшити 1000 features → 50 features для класифікації.

**Рішення:** PCA

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50)),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
```

---

### Use Case 5: Signal Separation (Cocktail Party)

**Завдання:** Розділити змішані аудіо сигнали.

**Рішення:** ICA

```python
from sklearn.decomposition import FastICA

# Mixed signals
X_mixed = ...  # shape: (n_samples, n_signals)

# ICA
ica = FastICA(n_components=n_signals, random_state=42)
S_recovered = ica.fit_transform(X_mixed)
```

---

### Use Case 6: Face Recognition

**Завдання:** Розпізнавання облич.

**Рішення:** PCA (Eigenfaces) або NMF

```python
from sklearn.decomposition import PCA

# Training faces
X_faces = ...  # shape: (n_faces, n_pixels)

# PCA (Eigenfaces)
pca = PCA(n_components=150)  # Keep 150 eigenfaces
X_reduced = pca.fit_transform(X_faces)

# Classification
from sklearn.svm import SVC
classifier = SVC(kernel='rbf')
classifier.fit(X_reduced, y_labels)
```

---

## Практичні Приклади

### Приклад 1: MovieLens Dataset

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load MovieLens data
ratings = pd.read_csv('ratings.csv')  # user_id, movie_id, rating, timestamp

# Create user-item matrix
n_users = ratings['user_id'].nunique()
n_movies = ratings['movie_id'].nunique()

# Train/test split
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

# Create sparse matrix
from scipy.sparse import coo_matrix

R_train = coo_matrix(
    (train['rating'], (train['user_id'], train['movie_id'])),
    shape=(n_users, n_movies)
).toarray()

# Train model
model = MatrixFactorizationALS(n_factors=50, n_iterations=20)
model.fit(R_train)

# Evaluate on test
predictions = []
actuals = []
for _, row in test.iterrows():
    u, i, rating = int(row['user_id']), int(row['movie_id']), row['rating']
    pred = model.predict(u, i)
    predictions.append(pred)
    actuals.append(rating)

rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))
print(f"Test RMSE: {rmse:.4f}")
```

---

### Приклад 2: Image Denoising

```python
from skimage import data, util
import numpy as np

# Load noisy image
img_clean = data.camera()
img_noisy = util.random_noise(img_clean, mode='gaussian', var=0.01)

# SVD denoising
U, s, Vt = np.linalg.svd(img_noisy, full_matrices=False)

# Keep only top-k singular values
k = 50
s_denoised = s.copy()
s_denoised[k:] = 0

# Reconstruct
img_denoised = U @ np.diag(s_denoised) @ Vt

# Compare
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img_clean, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(img_noisy, cmap='gray')
axes[1].set_title('Noisy')
axes[1].axis('off')

axes[2].imshow(img_denoised, cmap='gray')
axes[2].set_title(f'Denoised (k={k})')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

---

### Приклад 3: Text Clustering

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans

# Load data
newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
documents = newsgroups.data[:1000]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(documents)

# NMF
nmf = NMF(n_components=20, random_state=42)
doc_topics = nmf.fit_transform(X)

# Clustering
kmeans = KMeans(n_clusters=20, random_state=42)
clusters = kmeans.fit_predict(doc_topics)

# Print sample from each cluster
for cluster_id in range(5):
    docs_in_cluster = [i for i, c in enumerate(clusters) if c == cluster_id]
    print(f"\n=== Cluster {cluster_id} ===")
    print(documents[docs_in_cluster[0]][:200])
```

---

## Додаткові Ресурси

### Papers (Must-Read)

1. **SVD і PCA:**
   - Jolliffe, I. T. (2002). "Principal Component Analysis"

2. **Matrix Factorization для CF:**
   - Koren, Y., Bell, R., & Volinsky, C. (2009). "Matrix Factorization Techniques for Recommender Systems"
   - Hu, Y., Koren, Y., & Volinsky, C. (2008). "Collaborative Filtering for Implicit Feedback Datasets"

3. **Advanced CF:**
   - Koren, Y. (2008). "Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model"
   - Koren, Y. (2009). "Collaborative Filtering with Temporal Dynamics"

4. **NMF:**
   - Lee, D. D., & Seung, H. S. (1999). "Learning the parts of objects by non-negative matrix factorization"

5. **Deep Learning CF:**
   - He, X., et al. (2017). "Neural Collaborative Filtering"

### Online Resources

- **Sklearn Documentation**: https://scikit-learn.org/stable/modules/decomposition.html
- **Surprise Library**: http://surpriselib.com/
- **Spark MLlib**: https://spark.apache.org/docs/latest/ml-collaborative-filtering.html

### Datasets

- **MovieLens**: https://grouplens.org/datasets/movielens/
- **Netflix Prize**: (historical)
- **Amazon Reviews**: http://jmcauley.ucsd.edu/data/amazon/
- **Last.fm**: https://www.last.fm/api

---

## Висновок

**Matrix Factorization** — це не один алгоритм, а ціла **родина методів**:

### Ключові Takeaways:

1. **Exact Methods (SVD, QR, LU, Cholesky)**
   - Математично точні
   - Для повних матриць
   - Базові building blocks

2. **Statistical Methods (PCA, Factor Analysis, ICA)**
   - Фокус на interpretability
   - Feature extraction
   - Dimensionality reduction

3. **ML Methods (GD MF, ALS, NMF, PMF, NCF)**
   - Для sparse даних
   - Recommender systems
   - Iterative optimization

4. **Вибір методу** залежить від:
   - Типу даних (sparse/dense)
   - Розміру даних
   - Вимог до interpretability
   - Обчислювальних ресурсів
   - Use case

### Загальна Схема:

```
Математика (SVD) 
    ↓
Статистика (PCA, ICA) 
    ↓
Machine Learning (MF для CF)
    ↓
Deep Learning (NCF, Autoencoders)
```

Кожен рівень **базується** на попередньому, але **адаптує** під конкретні потреби!

---

**Автор**: Claude  
**Дата**: 2025  
**Версія**: 2.0
