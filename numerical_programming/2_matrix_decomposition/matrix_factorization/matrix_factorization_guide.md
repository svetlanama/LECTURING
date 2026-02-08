# Matrix Factorization: Повний Гайд

## Зміст

1. [Вступ](#вступ)
2. [Основний Алгоритм](#основний-алгоритм)
3. [Процес Факторизації](#процес-факторизації)
4. [Feature Interpretation (Latent Factor Analysis)](#feature-interpretation-latent-factor-analysis)
5. [Практичне Застосування](#практичне-застосування)
6. [Приклади Коду](#приклади-коду)
7. [Інструменти та Бібліотеки](#інструменти-та-бібліотеки)

---

## Вступ

**Matrix Factorization** — це техніка машинного навчання, яка розкладає велику матрицю на добуток двох менших матриць. Найчастіше використовується в рекомендаційних системах (Netflix, Spotify, Amazon).

### Основна Ідея

```
R (m×n) ≈ U (m×k) × M (k×n)
```

Де:
- **R** — матриця рейтингів (користувачі × товари)
- **U** — матриця користувачів (користувачі × латентні features)
- **M** — матриця товарів (латентні features × товари)
- **k** — кількість латентних features (набагато менше ніж m або n)

### Приклад

```
     M1  M2  M3  M4  M5
A    3   1   1   3   1
B    1   2   4   1   3      ≈    U (4×2) × M (2×5)
C    3   1   1   3   1
D    4   3   5   4   4
```

---

## Основний Алгоритм

### 1. Постановка Задачі

**Мета**: Знайти матриці U і M, які мінімізують помилку реконструкції.

**Функція втрат (Loss Function)**:

```
L = Σ(r_ij - û_ij)² + λ(||U||² + ||M||²)
```

Де:
- `r_ij` — реальний рейтинг користувача i для товару j
- `û_ij = U[i] · M[j]` — передбачений рейтинг
- `λ` — регуляризаційний коефіцієнт (запобігає overfitting)

### 2. Методи Оптимізації

#### A. Gradient Descent (Градієнтний Спуск)

**Алгоритм:**

```
1. Ініціалізація: U, M = random values
2. For each iteration:
   a. Для кожного відомого рейтингу r_ij:
      - Обчислити помилку: e_ij = r_ij - (U[i] · M[j])
      - Оновити U[i]: U[i] += α(2·e_ij·M[j] - λ·U[i])
      - Оновити M[j]: M[j] += α(2·e_ij·U[i] - λ·M[j])
3. Повторювати до збіжності
```

**Параметри:**
- `α` (alpha) — learning rate (швидкість навчання)
- `λ` (lambda) — регуляризація
- `k` — кількість латентних features
- `iterations` — кількість ітерацій

#### B. Alternating Least Squares (ALS)

**Алгоритм:**

```
1. Ініціалізація: U, M = random values
2. For each iteration:
   a. Зафіксувати M, оптимізувати U
   b. Зафіксувати U, оптимізувати M
3. Повторювати до збіжності
```

**Переваги ALS:**
- Легко паралелізується
- Добре працює з розрідженими даними
- Використовується в Apache Spark MLlib

#### C. Stochastic Gradient Descent (SGD)

**Алгоритм:**

```
1. Ініціалізація: U, M = random values
2. For each epoch:
   a. Перемішати дані
   b. Для кожного рейтингу (випадково):
      - Обчислити градієнт
      - Оновити U[i] та M[j]
3. Повторювати
```

**Переваги SGD:**
- Швидше для великих датасетів
- Менше пам'яті
- Може уникнути локальних мінімумів

### 3. Математичні Деталі

#### Обчислення Градієнтів

Для користувача i та товару j:

```
∂L/∂U[i,f] = -2·e_ij·M[f,j] + 2λ·U[i,f]
∂L/∂M[f,j] = -2·e_ij·U[i,f] + 2λ·M[f,j]
```

Де `e_ij = r_ij - (U[i] · M[j])`

#### Update Rules

```
U[i,f] ← U[i,f] - α·∂L/∂U[i,f]
M[f,j] ← M[f,j] - α·∂L/∂M[f,j]
```

Спрощено:

```
U[i,f] ← U[i,f] + α(2·e_ij·M[f,j] - 2λ·U[i,f])
M[f,j] ← M[f,j] + α(2·e_ij·U[i,f] - 2λ·M[f,j])
```

---

## Процес Факторизації

### Крок 1: Підготовка Даних

```python
import numpy as np

# Матриця рейтингів (0 = немає рейтингу)
R = np.array([
    [3, 1, 1, 3, 1],
    [1, 2, 4, 1, 3],
    [3, 1, 1, 3, 1],
    [4, 3, 5, 4, 4]
])

# Маска для відомих рейтингів
known_mask = R > 0
```

### Крок 2: Ініціалізація

```python
# Параметри
num_users, num_items = R.shape
k = 2  # кількість латентних features
alpha = 0.002  # learning rate
lambda_reg = 0.02  # регуляризація
iterations = 5000

# Випадкова ініціалізація
np.random.seed(42)
U = np.random.rand(num_users, k)
M = np.random.rand(k, num_items)
```

### Крок 3: Навчання

```python
def matrix_factorization(R, k, alpha=0.002, lambda_reg=0.02, iterations=5000):
    """
    Matrix Factorization using Gradient Descent
    
    Parameters:
    -----------
    R : numpy.array
        Ratings matrix (users × items)
    k : int
        Number of latent features
    alpha : float
        Learning rate
    lambda_reg : float
        Regularization parameter
    iterations : int
        Number of iterations
        
    Returns:
    --------
    U : numpy.array
        User features matrix (users × k)
    M : numpy.array
        Item features matrix (k × items)
    """
    m, n = R.shape
    
    # Ініціалізація
    U = np.random.rand(m, k)
    M = np.random.rand(k, n)
    
    # Навчання
    for step in range(iterations):
        # Для кожного користувача та товару
        for i in range(m):
            for j in range(n):
                if R[i, j] > 0:  # Тільки для відомих рейтингів
                    # Передбачення
                    prediction = np.dot(U[i, :], M[:, j])
                    
                    # Помилка
                    error = R[i, j] - prediction
                    
                    # Градієнтний спуск
                    for f in range(k):
                        # Оновлення U
                        U[i, f] += alpha * (2 * error * M[f, j] - lambda_reg * U[i, f])
                        
                        # Оновлення M
                        M[f, j] += alpha * (2 * error * U[i, f] - lambda_reg * M[f, j])
        
        # Обчислення загальної помилки (кожні 1000 ітерацій)
        if step % 1000 == 0:
            total_error = 0
            for i in range(m):
                for j in range(n):
                    if R[i, j] > 0:
                        prediction = np.dot(U[i, :], M[:, j])
                        total_error += (R[i, j] - prediction) ** 2
                        
            # Додаємо регуляризацію
            total_error += lambda_reg * (np.sum(U**2) + np.sum(M**2))
            
            print(f"Iteration {step}: Error = {total_error:.4f}")
    
    return U, M

# Запуск
U, M = matrix_factorization(R, k=2)
```

### Крок 4: Реконструкція

```python
# Передбачені рейтинги
R_predicted = np.dot(U, M)

print("Оригінальна матриця:")
print(R)
print("\nПередбачена матриця:")
print(R_predicted.round(2))
```

### Крок 5: Оцінка Якості

```python
from sklearn.metrics import mean_squared_error

# MSE для відомих рейтингів
known_ratings = R[R > 0]
predicted_ratings = R_predicted[R > 0]
mse = mean_squared_error(known_ratings, predicted_ratings)
rmse = np.sqrt(mse)

print(f"RMSE: {rmse:.4f}")
```

---

## Feature Interpretation (Latent Factor Analysis)

### Що Таке Латентні Features?

**Латентні features** — це приховані характеристики, які алгоритм виявляє автоматично, аналізуючи паттерни в даних.

**ВАЖЛИВО**: Алгоритм НЕ знає що це за features! Він знаходить лише математичні паттерни.

### Процес Інтерпретації

#### 1. Аналіз Матриці M (Item Features)

```python
# Після навчання маємо:
M = np.array([
    [1.4, 3.1, 0.3, 2.5, 0.2],  # Feature 0
    [2.5, 1.5, 4.4, 0.4, 1.1]   # Feature 1
])

# Візуалізуємо
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

# Feature 0
plt.subplot(1, 2, 1)
plt.bar(range(5), M[0])
plt.title('Feature 0 Distribution')
plt.xlabel('Movies')
plt.ylabel('Feature Value')
plt.xticks(range(5), ['M1', 'M2', 'M3', 'M4', 'M5'])

# Feature 1
plt.subplot(1, 2, 2)
plt.bar(range(5), M[1])
plt.title('Feature 1 Distribution')
plt.xlabel('Movies')
plt.ylabel('Feature Value')
plt.xticks(range(5), ['M1', 'M2', 'M3', 'M4', 'M5'])

plt.tight_layout()
plt.show()
```

#### 2. Top-N Аналіз

```python
def interpret_features(M, movie_metadata, top_n=3):
    """
    Інтерпретація латентних features
    
    Parameters:
    -----------
    M : numpy.array
        Item features matrix
    movie_metadata : dict
        Metadata про фільми {movie_id: {'genre': ..., 'director': ...}}
    top_n : int
        Кількість топ фільмів для аналізу
    """
    k, num_items = M.shape
    
    for feature_idx in range(k):
        print(f"\n{'='*50}")
        print(f"Feature {feature_idx} Analysis")
        print(f"{'='*50}")
        
        # Топ фільми для цього feature
        top_items = np.argsort(M[feature_idx])[-top_n:][::-1]
        
        print(f"\nTop {top_n} movies:")
        for rank, item_idx in enumerate(top_items, 1):
            value = M[feature_idx, item_idx]
            metadata = movie_metadata.get(f'M{item_idx+1}', {})
            print(f"  {rank}. M{item_idx+1} (value: {value:.2f})")
            print(f"     Genre: {metadata.get('genre', 'Unknown')}")
            print(f"     Director: {metadata.get('director', 'Unknown')}")
        
        # Аналіз спільних характеристик
        genres = [movie_metadata.get(f'M{i+1}', {}).get('genre') 
                  for i in top_items]
        directors = [movie_metadata.get(f'M{i+1}', {}).get('director') 
                     for i in top_items]
        
        # Визначаємо найчастіші
        from collections import Counter
        genre_counts = Counter(genres)
        director_counts = Counter(directors)
        
        print(f"\nPattern Analysis:")
        print(f"  Most common genre: {genre_counts.most_common(1)}")
        print(f"  Most common director: {director_counts.most_common(1)}")

# Приклад використання
movie_metadata = {
    'M1': {'genre': 'Action', 'director': 'Nolan', 'year': 2010},
    'M2': {'genre': 'Comedy', 'director': 'Apatow', 'year': 2012},
    'M3': {'genre': 'Action', 'director': 'Bay', 'year': 2015},
    'M4': {'genre': 'Comedy', 'director': 'Apatow', 'year': 2018},
    'M5': {'genre': 'Action', 'director': 'Bay', 'year': 2020}
}

interpret_features(M, movie_metadata)
```

#### 3. Кореляційний Аналіз

```python
def correlation_analysis(M, movie_metadata):
    """
    Аналіз кореляції features з metadata
    """
    from scipy.stats import pearsonr
    
    k, num_items = M.shape
    
    # Кодуємо metadata
    genres = [movie_metadata.get(f'M{i+1}', {}).get('genre', 'Unknown') 
              for i in range(num_items)]
    
    # Перетворюємо жанри в числа (наприклад, Comedy=1, Action=0)
    genre_vector = [1 if g == 'Comedy' else 0 for g in genres]
    
    # Обчислюємо кореляції
    for feature_idx in range(k):
        corr, p_value = pearsonr(M[feature_idx], genre_vector)
        print(f"Feature {feature_idx} correlation with Comedy genre: "
              f"{corr:.3f} (p-value: {p_value:.4f})")
        
        if abs(corr) > 0.7:
            if corr > 0:
                print(f"  → Feature {feature_idx} likely represents Comedy")
            else:
                print(f"  → Feature {feature_idx} likely represents Action")

correlation_analysis(M, movie_metadata)
```

#### 4. Візуалізація Feature Space

```python
def visualize_feature_space(M, movie_metadata):
    """
    2D візуалізація features
    """
    import matplotlib.pyplot as plt
    
    if M.shape[0] == 2:
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        for i in range(M.shape[1]):
            metadata = movie_metadata.get(f'M{i+1}', {})
            genre = metadata.get('genre', 'Unknown')
            color = 'red' if genre == 'Comedy' else 'blue'
            
            plt.scatter(M[0, i], M[1, i], c=color, s=200, alpha=0.6)
            plt.annotate(f'M{i+1}', (M[0, i], M[1, i]), 
                        fontsize=12, ha='center')
        
        plt.xlabel('Feature 0', fontsize=12)
        plt.ylabel('Feature 1', fontsize=12)
        plt.title('Movie Feature Space', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.6, label='Comedy'),
            Patch(facecolor='blue', alpha=0.6, label='Action')
        ]
        plt.legend(handles=legend_elements)
        
        plt.show()

visualize_feature_space(M, movie_metadata)
```

### Методи Інтерпретації

| Метод | Опис | Коли Використовувати |
|-------|------|---------------------|
| **Top-N Analysis** | Аналіз топ елементів для кожного feature | Початковий exploratory аналіз |
| **Correlation Analysis** | Кореляція з metadata | Є додаткові дані про items |
| **Clustering** | Кластеризація items за features | Виявлення груп схожих items |
| **Visualization** | 2D/3D візуалізація | k=2 або k=3 features |
| **A/B Testing** | Перевірка гіпотез про features | Production система |

### Можливі Інтерпретації

Features можуть представляти:

#### У Фільмах:
- Жанр (Comedy, Action, Drama)
- Режисер (Nolan, Spielberg)
- Епоха (80s, 90s, 2000s)
- Настрій (Light, Serious)
- Складність сюжету
- Візуальний стиль
- Цільова аудиторія

#### У Музиці:
- Жанр (Rock, Pop, Jazz)
- Темп (BPM)
- Енергійність
- Настрій (Happy, Sad)
- Вокал vs Інструментал
- Епоха запису

#### У E-commerce:
- Категорія товару
- Ціновий сегмент
- Бренд
- Цільова аудиторія
- Сезонність

### Валідація Інтерпретації

```python
def validate_interpretation(U, M, R, feature_names):
    """
    Перевірка якості інтерпретації
    """
    # 1. Перевірка consistency
    # Чи схожі користувачі дають схожі рейтинги?
    
    # 2. Перевірка predictability
    # Чи покращує інтерпретація передбачення?
    
    # 3. Human evaluation
    # Показати експертам і отримати feedback
    
    print("Interpretation Validation:")
    print(f"Feature names: {feature_names}")
    print(f"User preferences match expected patterns: Check manually")
    print(f"Recommendations make sense: A/B test in production")
```

---

## Практичне Застосування

### 1. Рекомендаційні Системи

#### Передбачення Рейтингу

```python
def predict_rating(user_id, item_id, U, M):
    """
    Передбачити рейтинг користувача для товару
    """
    return np.dot(U[user_id], M[:, item_id])

# Приклад
user_a_rating_for_new_movie = predict_rating(0, 2, U, M)
print(f"User A predicted rating for M3: {user_a_rating_for_new_movie:.2f}")
```

#### Top-N Рекомендації

```python
def recommend_top_n(user_id, U, M, R, n=5, exclude_rated=True):
    """
    Рекомендувати топ N товарів для користувача
    
    Parameters:
    -----------
    user_id : int
        ID користувача
    U : numpy.array
        User features matrix
    M : numpy.array
        Item features matrix
    R : numpy.array
        Original ratings matrix
    n : int
        Кількість рекомендацій
    exclude_rated : bool
        Виключити вже оцінені товари
    """
    # Передбачити всі рейтинги
    predicted_ratings = np.dot(U[user_id], M)
    
    if exclude_rated:
        # Виключити вже оцінені
        predicted_ratings[R[user_id] > 0] = -np.inf
    
    # Топ N
    top_items = np.argsort(predicted_ratings)[-n:][::-1]
    
    print(f"Top {n} recommendations for User {user_id}:")
    for rank, item_id in enumerate(top_items, 1):
        rating = predicted_ratings[item_id]
        print(f"  {rank}. Item {item_id} (predicted rating: {rating:.2f})")
    
    return top_items

# Приклад
recommend_top_n(user_id=0, U=U, M=M, R=R, n=3)
```

#### Similar Items

```python
def find_similar_items(item_id, M, top_n=5):
    """
    Знайти схожі товари
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Обчислити cosine similarity
    similarities = cosine_similarity(M.T)
    
    # Топ схожих (виключаючи сам товар)
    similar_items = np.argsort(similarities[item_id])[-top_n-1:-1][::-1]
    
    print(f"Items similar to Item {item_id}:")
    for rank, similar_id in enumerate(similar_items, 1):
        similarity = similarities[item_id, similar_id]
        print(f"  {rank}. Item {similar_id} (similarity: {similarity:.3f})")
    
    return similar_items

# Приклад
find_similar_items(item_id=0, M=M, top_n=3)
```

### 2. Cold Start Problem

#### New User

```python
def handle_new_user(initial_ratings, M, k):
    """
    Обробка нового користувача з кількома рейтингами
    
    Parameters:
    -----------
    initial_ratings : dict
        {item_id: rating} початкові рейтинги
    M : numpy.array
        Item features matrix
    k : int
        Number of features
    """
    # Ініціалізація user profile
    u_new = np.random.rand(k)
    
    # Оптимізація на основі початкових рейтингів
    alpha = 0.01
    for _ in range(100):
        for item_id, rating in initial_ratings.items():
            prediction = np.dot(u_new, M[:, item_id])
            error = rating - prediction
            u_new += alpha * 2 * error * M[:, item_id]
    
    return u_new

# Приклад
new_user_ratings = {0: 5, 2: 4}  # M1=5, M3=4
u_new = handle_new_user(new_user_ratings, M, k=2)
print(f"New user profile: {u_new}")
```

#### New Item

```python
def handle_new_item(initial_ratings, U, k):
    """
    Обробка нового товару з кількома рейтингами
    """
    # Аналогічно до new user
    m_new = np.random.rand(k)
    
    alpha = 0.01
    for _ in range(100):
        for user_id, rating in initial_ratings.items():
            prediction = np.dot(U[user_id], m_new)
            error = rating - prediction
            m_new += alpha * 2 * error * U[user_id]
    
    return m_new
```

### 3. Implicit Feedback

Для даних де немає явних рейтингів (лайки, перегляди, кліки):

```python
def implicit_matrix_factorization(interactions, k, alpha=0.002, 
                                  lambda_reg=0.02, iterations=5000):
    """
    Matrix Factorization для implicit feedback
    
    Parameters:
    -----------
    interactions : numpy.array
        Binary matrix (1=interaction, 0=no interaction)
    """
    m, n = interactions.shape
    
    U = np.random.rand(m, k)
    M = np.random.rand(k, n)
    
    # Confidence weights
    confidence = 1 + 40 * interactions
    
    for step in range(iterations):
        # Оптимізація з врахуванням confidence
        for i in range(m):
            for j in range(n):
                prediction = np.dot(U[i, :], M[:, j])
                
                # Зважена помилка
                error = confidence[i, j] * (interactions[i, j] - prediction)
                
                for f in range(k):
                    U[i, f] += alpha * (2 * error * M[f, j] - lambda_reg * U[i, f])
                    M[f, j] += alpha * (2 * error * U[i, f] - lambda_reg * M[f, j])
        
        if step % 1000 == 0:
            total_loss = np.sum(confidence * (interactions - np.dot(U, M))**2)
            total_loss += lambda_reg * (np.sum(U**2) + np.sum(M**2))
            print(f"Step {step}: Loss = {total_loss:.4f}")
    
    return U, M
```

### 4. Temporal Dynamics

Врахування часу:

```python
def temporal_matrix_factorization(R, timestamps, k, alpha=0.002, 
                                  beta=0.01, iterations=5000):
    """
    Matrix Factorization з врахуванням часу
    
    Parameters:
    -----------
    timestamps : numpy.array
        Часові мітки для кожного рейтингу
    beta : float
        Time decay параметр
    """
    m, n = R.shape
    
    U = np.random.rand(m, k)
    M = np.random.rand(k, n)
    
    # Bias terms
    user_bias = np.zeros(m)
    item_bias = np.zeros(n)
    global_bias = np.mean(R[R > 0])
    
    current_time = np.max(timestamps)
    
    for step in range(iterations):
        for i in range(m):
            for j in range(n):
                if R[i, j] > 0:
                    # Time decay
                    time_weight = np.exp(-beta * (current_time - timestamps[i, j]))
                    
                    # Prediction з bias
                    prediction = (global_bias + user_bias[i] + item_bias[j] + 
                                 np.dot(U[i, :], M[:, j]))
                    
                    error = time_weight * (R[i, j] - prediction)
                    
                    # Updates
                    user_bias[i] += alpha * error
                    item_bias[j] += alpha * error
                    
                    for f in range(k):
                        U[i, f] += alpha * (error * M[f, j])
                        M[f, j] += alpha * (error * U[i, f])
    
    return U, M, user_bias, item_bias, global_bias
```

---

## Приклади Коду

### Повна Імплементація

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class MatrixFactorization:
    """
    Повна імплементація Matrix Factorization
    """
    
    def __init__(self, k=2, alpha=0.002, lambda_reg=0.02, 
                 iterations=5000, verbose=True):
        """
        Parameters:
        -----------
        k : int
            Number of latent features
        alpha : float
            Learning rate
        lambda_reg : float
            Regularization parameter
        iterations : int
            Number of training iterations
        verbose : bool
            Print training progress
        """
        self.k = k
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.iterations = iterations
        self.verbose = verbose
        
        self.U = None
        self.M = None
        self.training_errors = []
    
    def fit(self, R):
        """
        Навчити модель
        
        Parameters:
        -----------
        R : numpy.array
            Ratings matrix (users × items)
        """
        m, n = R.shape
        
        # Ініціалізація
        np.random.seed(42)
        self.U = np.random.rand(m, self.k)
        self.M = np.random.rand(self.k, n)
        
        # Навчання
        for step in range(self.iterations):
            # Gradient descent
            for i in range(m):
                for j in range(n):
                    if R[i, j] > 0:
                        prediction = np.dot(self.U[i, :], self.M[:, j])
                        error = R[i, j] - prediction
                        
                        # Update
                        for f in range(self.k):
                            self.U[i, f] += self.alpha * (
                                2 * error * self.M[f, j] - 
                                self.lambda_reg * self.U[i, f]
                            )
                            self.M[f, j] += self.alpha * (
                                2 * error * self.U[i, f] - 
                                self.lambda_reg * self.M[f, j]
                            )
            
            # Обчислити помилку
            if step % 100 == 0:
                total_error = 0
                for i in range(m):
                    for j in range(n):
                        if R[i, j] > 0:
                            prediction = np.dot(self.U[i, :], self.M[:, j])
                            total_error += (R[i, j] - prediction) ** 2
                
                # Регуляризація
                total_error += self.lambda_reg * (
                    np.sum(self.U**2) + np.sum(self.M**2)
                )
                
                self.training_errors.append(total_error)
                
                if self.verbose and step % 1000 == 0:
                    print(f"Iteration {step}: Error = {total_error:.4f}")
        
        return self
    
    def predict(self, user_id=None, item_id=None):
        """
        Передбачити рейтинги
        
        Parameters:
        -----------
        user_id : int, optional
            User ID (if None, predict for all users)
        item_id : int, optional
            Item ID (if None, predict for all items)
        
        Returns:
        --------
        predictions : numpy.array or float
            Predicted ratings
        """
        if user_id is not None and item_id is not None:
            return np.dot(self.U[user_id, :], self.M[:, item_id])
        elif user_id is not None:
            return np.dot(self.U[user_id, :], self.M)
        elif item_id is not None:
            return np.dot(self.U, self.M[:, item_id])
        else:
            return np.dot(self.U, self.M)
    
    def recommend(self, user_id, n=5, exclude_rated=None):
        """
        Рекомендувати топ N items
        """
        predictions = self.predict(user_id=user_id)
        
        if exclude_rated is not None:
            predictions[exclude_rated[user_id] > 0] = -np.inf
        
        top_items = np.argsort(predictions)[-n:][::-1]
        
        return top_items, predictions[top_items]
    
    def plot_training_curve(self):
        """
        Візуалізувати процес навчання
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_errors)
        plt.xlabel('Iteration (×100)')
        plt.ylabel('Total Error')
        plt.title('Training Curve')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def interpret_features(self, movie_names=None, feature_names=None):
        """
        Інтерпретувати латентні features
        """
        k, num_items = self.M.shape
        
        for f in range(k):
            feature_name = feature_names[f] if feature_names else f"Feature {f}"
            print(f"\n{feature_name}:")
            
            # Топ items
            top_items = np.argsort(self.M[f])[-5:][::-1]
            
            for rank, item_idx in enumerate(top_items, 1):
                item_name = movie_names[item_idx] if movie_names else f"Item {item_idx}"
                value = self.M[f, item_idx]
                print(f"  {rank}. {item_name}: {value:.3f}")

# Використання
if __name__ == "__main__":
    # Дані
    R = np.array([
        [3, 1, 1, 3, 1],
        [1, 2, 4, 1, 3],
        [3, 1, 1, 3, 1],
        [4, 3, 5, 4, 4]
    ])
    
    # Навчання
    mf = MatrixFactorization(k=2, iterations=5000)
    mf.fit(R)
    
    # Передбачення
    R_pred = mf.predict()
    print("\nPredicted Ratings:")
    print(R_pred.round(2))
    
    # RMSE
    known_mask = R > 0
    rmse = np.sqrt(mean_squared_error(R[known_mask], R_pred[known_mask]))
    print(f"\nRMSE: {rmse:.4f}")
    
    # Рекомендації
    print("\nRecommendations for User 0:")
    top_items, scores = mf.recommend(user_id=0, n=3, exclude_rated=R)
    for item, score in zip(top_items, scores):
        print(f"  Item {item}: {score:.2f}")
    
    # Training curve
    mf.plot_training_curve()
    
    # Інтерпретація
    movie_names = ['M1', 'M2', 'M3', 'M4', 'M5']
    mf.interpret_features(movie_names=movie_names)
```

### Використання Sklearn

```python
from sklearn.decomposition import NMF

# Non-negative Matrix Factorization
model = NMF(n_components=2, init='random', random_state=42, max_iter=500)

# Навчання
U = model.fit_transform(R)
M = model.components_

# Передбачення
R_pred = np.dot(U, M)

print("User Features:")
print(U)
print("\nItem Features:")
print(M)
print("\nReconstructed Matrix:")
print(R_pred.round(2))
```

### Використання Surprise Library

```python
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import pandas as pd

# Підготовка даних
ratings_dict = {
    'user': [],
    'item': [],
    'rating': []
}

for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if R[i, j] > 0:
            ratings_dict['user'].append(i)
            ratings_dict['item'].append(j)
            ratings_dict['rating'].append(R[i, j])

df = pd.DataFrame(ratings_dict)

# Створення датасету
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)

# SVD модель
algo = SVD(n_factors=2, n_epochs=20, lr_all=0.005, reg_all=0.02)

# Cross-validation
results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

# Навчання на всіх даних
trainset = data.build_full_trainset()
algo.fit(trainset)

# Передбачення
user_id = 0
item_id = 2
prediction = algo.predict(user_id, item_id)
print(f"Predicted rating: {prediction.est:.2f}")
```

---

## Інструменти та Бібліотеки

### Python Libraries

#### 1. **Scikit-learn**
```python
from sklearn.decomposition import NMF, TruncatedSVD

# NMF (Non-negative Matrix Factorization)
nmf = NMF(n_components=2, init='random', random_state=42)
U = nmf.fit_transform(R)
M = nmf.components_

# SVD
svd = TruncatedSVD(n_components=2, random_state=42)
U_svd = svd.fit_transform(R)
M_svd = svd.components_
```

**Переваги:**
- Простота використання
- Добре документовано
- Швидка імплементація

**Недоліки:**
- Обмежені можливості кастомізації
- Не підтримує розріджені матриці напряму

#### 2. **Surprise**
```bash
pip install scikit-surprise
```

```python
from surprise import SVD, NMF, KNNBasic
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, GridSearchCV

# Створення алгоритму
algo = SVD(n_factors=2, n_epochs=20, lr_all=0.005, reg_all=0.02)

# Grid search для підбору параметрів
param_grid = {
    'n_factors': [2, 5, 10],
    'n_epochs': [10, 20, 50],
    'lr_all': [0.001, 0.005, 0.01],
    'reg_all': [0.01, 0.02, 0.1]
}

gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
gs.fit(data)

print(f"Best RMSE: {gs.best_score['rmse']:.4f}")
print(f"Best params: {gs.best_params['rmse']}")
```

**Переваги:**
- Спеціалізується на рекомендаційних системах
- Багато алгоритмів (SVD, NMF, KNN, Slope One)
- Вбудований cross-validation та grid search

**Недоліки:**
- Менш гнучка для кастомних loss functions

#### 3. **TensorFlow/PyTorch**

**TensorFlow:**
```python
import tensorflow as tf

class MatrixFactorizationTF(tf.keras.Model):
    def __init__(self, num_users, num_items, k):
        super().__init__()
        self.user_embedding = tf.keras.layers.Embedding(num_users, k)
        self.item_embedding = tf.keras.layers.Embedding(num_items, k)
    
    def call(self, inputs):
        user_ids, item_ids = inputs
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        return tf.reduce_sum(user_vec * item_vec, axis=1)

# Створення моделі
model = MatrixFactorizationTF(num_users=4, num_items=5, k=2)

# Компіляція
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss='mse'
)

# Навчання
# Підготовка даних у форматі (user_ids, item_ids, ratings)
# model.fit([user_ids, item_ids], ratings, epochs=100)
```

**PyTorch:**
```python
import torch
import torch.nn as nn

class MatrixFactorizationPyTorch(nn.Module):
    def __init__(self, num_users, num_items, k):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, k)
        self.item_embedding = nn.Embedding(num_items, k)
        
        # Ініціалізація
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_ids, item_ids):
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        return (user_vec * item_vec).sum(dim=1)

# Використання
model = MatrixFactorizationPyTorch(num_users=4, num_items=5, k=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
# for epoch in range(100):
#     predictions = model(user_ids, item_ids)
#     loss = criterion(predictions, ratings)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
```

**Переваги:**
- Максимальна гнучкість
- GPU прискорення
- Легко додавати нові features (deep learning)

**Недоліки:**
- Більше boilerplate коду
- Потребує більше досвіду

#### 4. **LightFM**
```bash
pip install lightfm
```

```python
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score

# Створення моделі
model = LightFM(
    no_components=2,
    loss='warp',  # or 'bpr', 'logistic'
    learning_rate=0.05,
    random_state=42
)

# Навчання на sparse matrix
from scipy.sparse import csr_matrix
R_sparse = csr_matrix(R)
model.fit(R_sparse, epochs=30, num_threads=2)

# Передбачення
predictions = model.predict(user_ids, item_ids)

# Evaluation
precision = precision_at_k(model, R_sparse, k=5).mean()
print(f"Precision@5: {precision:.4f}")
```

**Переваги:**
- Підтримує hybrid recommendations (content + collaborative)
- Працює з sparse matrices
- Швидка імплементація на C

**Недоліки:**
- Менше документації
- Специфічний API

#### 5. **Implicit**
```bash
pip install implicit
```

```python
import implicit

# Для implicit feedback
from scipy.sparse import csr_matrix

# Створення sparse matrix
R_sparse = csr_matrix(R)

# ALS модель
model = implicit.als.AlternatingLeastSquares(
    factors=2,
    regularization=0.01,
    iterations=50
)

# Навчання
model.fit(R_sparse)

# Рекомендації
user_id = 0
recommendations = model.recommend(
    user_id, 
    R_sparse[user_id],
    N=5,
    filter_already_liked_items=True
)

print("Recommendations:")
for item_id, score in recommendations:
    print(f"  Item {item_id}: {score:.3f}")

# Similar items
similar_items = model.similar_items(item_id=0, N=5)
```

**Переваги:**
- Дуже швидка (Cython)
- Оптимізована для implicit feedback
- GPU підтримка

**Недоліки:**
- Тільки для implicit feedback
- Менше алгоритмів

### Big Data Tools

#### Apache Spark MLlib
```python
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

# Створення Spark session
spark = SparkSession.builder.appName("MatrixFactorization").getOrCreate()

# Підготовка даних
ratings_data = []
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if R[i, j] > 0:
            ratings_data.append((i, j, float(R[i, j])))

df = spark.createDataFrame(ratings_data, ["user", "item", "rating"])

# ALS модель
als = ALS(
    rank=2,
    maxIter=20,
    regParam=0.01,
    userCol="user",
    itemCol="item",
    ratingCol="rating",
    coldStartStrategy="drop"
)

# Навчання
model = als.fit(df)

# Рекомендації
user_recs = model.recommendForAllUsers(5)
user_recs.show()

# Передбачення
predictions = model.transform(df)
predictions.show()
```

**Використовується для:**
- Великі датасети (millions/billions ratings)
- Розподілене навчання
- Production systems

### Comparison Table

| Library | Best For | Pros | Cons | Difficulty |
|---------|----------|------|------|------------|
| **Scikit-learn** | Quick prototyping | Simple, fast | Limited customization | Easy |
| **Surprise** | Research, benchmarking | Many algorithms, good docs | Less flexible | Easy |
| **TensorFlow/PyTorch** | Custom models, deep learning | Maximum flexibility, GPU | More code | Hard |
| **LightFM** | Hybrid recommendations | Content + collaborative | Specific use case | Medium |
| **Implicit** | Implicit feedback, speed | Very fast, GPU support | Limited algorithms | Medium |
| **Spark MLlib** | Big data | Distributed, scalable | Overhead for small data | Hard |

---

## Додаткові Ресурси

### Статті та Papers

1. **Koren, Y., Bell, R., & Volinsky, C. (2009)**
   "Matrix Factorization Techniques for Recommender Systems"
   IEEE Computer, 42(8), 30-37.

2. **Hu, Y., Koren, Y., & Volinsky, C. (2008)**
   "Collaborative Filtering for Implicit Feedback Datasets"
   ICDM 2008.

3. **Rendle, S. (2010)**
   "Factorization Machines"
   ICDM 2010.

### Онлайн Курси

- **Coursera**: Recommender Systems Specialization
- **Fast.ai**: Collaborative Filtering
- **Udacity**: Building Recommendation Systems

### Книги

1. **"Recommender Systems Handbook"** by Ricci et al.
2. **"Mining of Massive Datasets"** by Leskovec, Rajaraman, Ullman
3. **"Programming Collective Intelligence"** by Toby Segaran

### Datasets для Практики

- **MovieLens**: Рейтинги фільмів (100K - 25M)
- **Amazon Product Reviews**: E-commerce
- **Last.fm**: Музичні дані
- **Yelp**: Ресторани та бізнеси
- **Book-Crossing**: Книги

### Корисні Посилання

- [Surprise Documentation](http://surpriselib.com/)
- [LightFM GitHub](https://github.com/lyst/lightfm)
- [Implicit GitHub](https://github.com/benfred/implicit)
- [Netflix Prize](https://www.netflixprize.com/)

---

## Висновок

Matrix Factorization — це потужна техніка для:
- Рекомендаційних систем
- Dimensionality reduction
- Collaborative filtering
- Feature learning

**Ключові моменти:**
1. Алгоритм знаходить латентні паттерни автоматично
2. Features потребують людської інтерпретації
3. Є багато варіацій (SVD, NMF, ALS, deep learning)
4. Вибір методу залежить від задачі та даних

**Next Steps:**
- Експериментуйте з різними k (кількість features)
- Пробуйте різні loss functions
- Додавайте bias terms
- Враховуйте час (temporal dynamics)
- Комбінуйте з deep learning (Neural Collaborative Filtering)

---

**Автор**: Claude  
**Дата**: 2025  
**Версія**: 1.0
