from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.linear_model import LogisticRegression
import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURACIÓN: Fracción de datos a utilizar
# ============================================================================
# Valores válidos: 0.0 < SAMPLE_FRACTION <= 1.0
# Ejemplo: 0.4 = usar 40% de los datos, 1.0 = usar todos los datos
SAMPLE_FRACTION = 0.5  # Cambiar este valor para reducir el volumen de datos

print(f"\n{'='*80}")
print(f"INICIANDO SCRIPT - SAMPLE_FRACTION = {SAMPLE_FRACTION}")
print(f"Directorio actual: {os.getcwd()}")
print(f"{'='*80}")

# Carga de datos
path = "dim_reduction/"
print(f"\nCargando datos desde: {os.path.abspath(path)}")

# Cargar datos de PCA
data = np.load(path + 'pca_reduced.npz')
X_train_pca = data['X_train']
y_train = data['y_train']
X_test_pca = data['X_test']
y_test = data['y_test']

# Cargar datos de SVD
data_svd = np.load(path + 'svd_reduced.npz')
X_train_svd = data_svd['X_train']
X_test_svd = data_svd['X_test']

# Cargar datos de FA
data_fa = np.load(path + 'fa_reduced.npz')
X_train_fa = data_fa['X_train']
X_test_fa = data_fa['X_test']

# Cargar datos de LDA
data_lda = np.load(path + 'lda_reduced.npz')
X_train_lda = data_lda['X_train']
X_test_lda = data_lda['X_test']

# Cargar datos de JL
data_jl = np.load(path + 'jl_reduced.npz')
X_train_jl = data_jl['X_train']
X_test_jl = data_jl['X_test']

# ============================================================================
# Aplicar muestreo si SAMPLE_FRACTION < 1.0
# ============================================================================
if SAMPLE_FRACTION < 1.0:
    print(f"\n{'='*80}")
    print(f"APLICANDO MUESTREO ESTRATIFICADO: {SAMPLE_FRACTION*100:.1f}% de los datos")
    print(f"{'='*80}")
    
    # Calcular índices de muestreo estratificado (mantiene proporción de clases)
    # Usamos el mismo random_state para asegurar consistencia
    train_indices = np.arange(len(y_train))
    sampled_train_indices, _ = train_test_split(
        train_indices,
        train_size=SAMPLE_FRACTION,
        stratify=y_train,
        random_state=42
    )
    
    test_indices = np.arange(len(y_test))
    sampled_test_indices, _ = train_test_split(
        test_indices,
        train_size=SAMPLE_FRACTION,
        stratify=y_test,
        random_state=42
    )
    
    # Aplicar el mismo muestreo a todos los métodos de reducción
    X_train_pca = X_train_pca[sampled_train_indices]
    X_test_pca = X_test_pca[sampled_test_indices]
    
    X_train_svd = X_train_svd[sampled_train_indices]
    X_test_svd = X_test_svd[sampled_test_indices]
    
    X_train_fa = X_train_fa[sampled_train_indices]
    X_test_fa = X_test_fa[sampled_test_indices]
    
    X_train_lda = X_train_lda[sampled_train_indices]
    X_test_lda = X_test_lda[sampled_test_indices]
    
    X_train_jl = X_train_jl[sampled_train_indices]
    X_test_jl = X_test_jl[sampled_test_indices]
    
    # Aplicar a las etiquetas (solo una vez)
    y_train = y_train[sampled_train_indices]
    y_test = y_test[sampled_test_indices]
    
    print(f"\n✓ Muestreo completado:")
    print(f"  Train: {len(y_train)} muestras (clases: {np.bincount(y_train)})")
    print(f"  Test: {len(y_test)} muestras (clases: {np.bincount(y_test)})")
    print(f"{'='*80}\n")
else:
    print(f"\n{'='*80}")
    print(f"USANDO DATASET COMPLETO (SAMPLE_FRACTION = {SAMPLE_FRACTION})")
    print(f"  Train: {len(y_train)} muestras")
    print(f"  Test: {len(y_test)} muestras")
    print(f"{'='*80}\n")

# Diccionario con todos los métodos de reducción (después del muestreo)
reduction_methods = {
    'pca': (X_train_pca, X_test_pca),
    'svd': (X_train_svd, X_test_svd),
    'fa': (X_train_fa, X_test_fa),
    'lda': (X_train_lda, X_test_lda),
    'jl': (X_train_jl, X_test_jl)
}

# Crear directorios para los modelos
models_dir = "trained_models"
os.makedirs(models_dir, exist_ok=True)
for subdir in ['svm', 'random_forest', 'logistic_regression']:
    os.makedirs(os.path.join(models_dir, subdir), exist_ok=True)

def check_model_exists(model_type, method_name, params_dict):
    """
    Verifica si un modelo ya fue entrenado previamente.
    
    Args:
        model_type: 'svm', 'random_forest', 'logistic_regression'
        method_name: 'pca', 'svd', 'fa', 'lda', 'jl'
        params_dict: diccionario con parámetros del modelo (ej: {'C': 10, 'n_estimators': 100})
    
    Returns:
        tuple: (exists: bool, model_path: str)
    """
    if model_type == 'svm':
        C = params_dict.get('C', 10)
        filename = f'svm_C{C}_{method_name}.pkl'
    elif model_type == 'random_forest':
        n = params_dict.get('n_estimators', 100)
        filename = f'rf_n{n}_{method_name}.pkl'
    elif model_type == 'logistic_regression':
        filename = f'logistic_{method_name}.pkl'
    else:
        return False, ""
    
    model_path = os.path.join(models_dir, model_type, filename)
    return os.path.exists(model_path), model_path

def load_existing_results(results_pattern='training_results_*.csv'):
    """
    Carga resultados de entrenamientos previos si existen.
    
    Returns:
        list: lista de diccionarios con resultados previos
    """
    import glob
    
    csv_files = sorted(glob.glob(results_pattern), reverse=True)
    if not csv_files:
        return []
    
    # Cargar el CSV más reciente
    try:
        df = pd.read_csv(csv_files[0])
        print(f"Cargados {len(df)} resultados previos de: {csv_files[0]}")
        return df.to_dict('records')
    except Exception as e:
        print(f"No se pudieron cargar resultados previos: {e}")
        return []

def showResults(y_test, y_pred, model_name, reduction_method):
    print(f"\n{'='*60}")
    print(f"RESULTADOS: {model_name.upper()} + {reduction_method.upper()}")
    print(f"{'='*60}")
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['IA', 'Humano'], digits=4))

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nInterpretación:")
    print(f"  TN (IA correctamente clasificada): {cm[0,0]}")
    print(f"  FP (IA clasificada como Humano): {cm[0,1]}")
    print(f"  FN (Humano clasificado como IA): {cm[1,0]}")
    print(f"  TP (Humano correctamente clasificado): {cm[1,1]}")
    
    return {
        'model': model_name,
        'reduction_method': reduction_method,
        'accuracy': accuracy,
        'f1_score': f1,
        'tn': cm[0,0],
        'fp': cm[0,1],
        'fn': cm[1,0],
        'tp': cm[1,1]
    }


# Lista para almacenar todos los resultados
all_results = load_existing_results()

print(f"\n{'#'*80}")
print("ENTRENAMIENTO DE MODELOS CON TODOS LOS MÉTODOS DE REDUCCIÓN")
print(f"{'#'*80}")
if all_results:
    print(f"MODO RESUMPTION: Se encontraron {len(all_results)} resultados previos")
    print(f"Se saltarán los modelos ya entrenados\n")
else:
    print(f"Iniciando entrenamiento desde cero\n")

# ============================================================================
# 1. SVM
# ============================================================================
print(f"\n{'='*80}")
print("1. SUPPORT VECTOR MACHINE (SVM)")
print(f"{'='*80}")

C = 10
for method_name, (X_train_method, X_test_method) in reduction_methods.items():
    # Verificar si el modelo ya existe
    exists, model_path = check_model_exists('svm', method_name, {'C': C})
    
    if exists:
        print(f"\n--- SALTANDO SVM con {method_name.upper()} (ya existe: {model_path}) ---")
        continue
    
    print(f"\n--- Entrenando SVM con {method_name.upper()} ---")
    
    svm_model = SVC(kernel='rbf', C=C, gamma='scale', random_state=42)
    svm_model.fit(X_train_method, y_train)
    y_pred = svm_model.predict(X_test_method)
    
    # Mostrar y guardar resultados
    results = showResults(y_test, y_pred, 'svm', method_name)
    all_results.append(results)
    
    # Guardar modelo
    model_filename = f'svm_C{C}_{method_name}.pkl'
    model_path = os.path.join(models_dir, 'svm', model_filename)
    joblib.dump(svm_model, model_path)
    print(f"Modelo guardado: {model_path}")
    
    # Guardar resultados intermedios después de cada modelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f'training_results_{timestamp}.csv'
    pd.DataFrame(all_results).to_csv(results_filename, index=False)
    print(f"Resultados guardados (checkpoint): {results_filename}")

# ============================================================================
# 2. Random Forest
# ============================================================================
print(f"\n{'='*80}")
print("2. RANDOM FOREST")
print(f"{'='*80}")

n_estimators = 100
for method_name, (X_train_method, X_test_method) in reduction_methods.items():
    # Verificar si el modelo ya existe
    exists, model_path = check_model_exists('random_forest', method_name, {'n_estimators': n_estimators})
    
    if exists:
        print(f"\n--- SALTANDO Random Forest con {method_name.upper()} (ya existe: {model_path}) ---")
        continue
    
    print(f"\n--- Entrenando Random Forest con {method_name.upper()} ---")
    
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train_method, y_train)
    y_pred = rf_model.predict(X_test_method)
    
    # Mostrar y guardar resultados
    results = showResults(y_test, y_pred, 'random_forest', method_name)
    all_results.append(results)
    
    # Guardar modelo
    model_filename = f'rf_n{n_estimators}_{method_name}.pkl'
    model_path = os.path.join(models_dir, 'random_forest', model_filename)
    joblib.dump(rf_model, model_path)
    print(f"✓ Modelo guardado: {model_path}")
    
    # Guardar resultados intermedios después de cada modelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f'training_results_{timestamp}.csv'
    pd.DataFrame(all_results).to_csv(results_filename, index=False)
    print(f"Resultados guardados (checkpoint): {results_filename}")

# ============================================================================
# 3. Logistic Regression
# ============================================================================
print(f"\n{'='*80}")
print("3. LOGISTIC REGRESSION")
print(f"{'='*80}")

for method_name, (X_train_method, X_test_method) in reduction_methods.items():
    # Verificar si el modelo ya existe
    exists, model_path = check_model_exists('logistic_regression', method_name, {})
    
    if exists:
        print(f"\n--- SALTANDO Logistic Regression con {method_name.upper()} (ya existe: {model_path}) ---")
        continue
    
    print(f"\n--- Entrenando Logistic Regression con {method_name.upper()} ---")
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_method, y_train)
    y_pred = lr_model.predict(X_test_method)
    
    # Mostrar y guardar resultados
    results = showResults(y_test, y_pred, 'logistic_regression', method_name)
    all_results.append(results)
    
    # Guardar modelo
    model_filename = f'logistic_{method_name}.pkl'
    model_path = os.path.join(models_dir, 'logistic_regression', model_filename)
    joblib.dump(lr_model, model_path)
    print(f"✓ Modelo guardado: {model_path}")
    
    # Guardar resultados intermedios después de cada modelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f'training_results_{timestamp}.csv'
    pd.DataFrame(all_results).to_csv(results_filename, index=False)
    print(f"Resultados guardados (checkpoint): {results_filename}")

# ============================================================================
# Resumen de resultados
# ============================================================================
print(f"\n{'#'*80}")
print("RESUMEN DE RESULTADOS")
print(f"{'#'*80}\n")

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('accuracy', ascending=False)

print(results_df.to_string(index=False))

# Guardar resultados en CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_filename = f'training_results_{timestamp}.csv'
results_df.to_csv(results_filename, index=False)
print(f"\n✓ Resultados guardados en: {results_filename}")

# Mostrar mejor modelo por métrica
print(f"\n{'='*60}")
print("MEJORES MODELOS")
print(f"{'='*60}")
best_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
best_f1 = results_df.loc[results_df['f1_score'].idxmax()]

print(f"\nMejor Accuracy: {best_accuracy['accuracy']:.4f}")
print(f"  Modelo: {best_accuracy['model']} + {best_accuracy['reduction_method']}")

print(f"\nMejor F1-Score: {best_f1['f1_score']:.4f}")
print(f"  Modelo: {best_f1['model']} + {best_f1['reduction_method']}")

print(f"\n{'#'*80}")
print("ENTRENAMIENTO COMPLETADO")
print(f"{'#'*80}")