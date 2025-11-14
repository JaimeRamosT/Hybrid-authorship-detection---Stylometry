"""
Ejemplo de cómo cargar y trabajar con archivos parciales en un notebook o script
"""
import pandas as pd
import glob
from pathlib import Path

# ============================================================================
# EJEMPLO 1: Cargar el dataframe completo combinado
# ============================================================================
def load_combined_dataframe(partials_dir="data/features_partials"):
    """Carga el archivo combinado si existe"""
    combined_path = Path(partials_dir) / "features_combined.parquet"
    
    if combined_path.exists():
        print(f"[INFO] Cargando archivo combinado: {combined_path}")
        df = pd.read_parquet(combined_path)
        print(f"[✓] Cargado: {len(df)} filas, {len(df.columns)} columnas")
        return df
    else:
        print(f"[ERROR] Archivo combinado no encontrado en: {combined_path}")
        print(f"[INFO] Ejecuta primero: python combine_partials.py --partials-dir '{partials_dir}'")
        return None

# ============================================================================
# EJEMPLO 2: Combinar manualmente todos los parciales
# ============================================================================
def combine_all_partials(partials_dir="data/features_partials"):
    """Combina todos los archivos parciales en memoria"""
    files = sorted(glob.glob(str(Path(partials_dir) / "partial_*.parquet")) + 
                   glob.glob(str(Path(partials_dir) / "partial_*.csv.gz")))
    
    if not files:
        print(f"[ERROR] No se encontraron archivos parciales en: {partials_dir}")
        return None
    
    print(f"[INFO] Encontrados {len(files)} archivos parciales")
    
    dfs = []
    for i, f in enumerate(files, 1):
        print(f"[LOAD] ({i}/{len(files)}) {Path(f).name}", end=" ")
        if f.endswith(".parquet"):
            df = pd.read_parquet(f)
        else:
            df = pd.read_csv(f, compression='gzip')
        print(f"- {len(df)} filas")
        dfs.append(df)
    
    print(f"\n[INFO] Concatenando...")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"[✓] Total: {len(combined)} filas, {len(combined.columns)} columnas")
    
    return combined

# ============================================================================
# EJEMPLO 3: Cargar SOLO algunos parciales específicos (subset)
# ============================================================================
def load_partial_subset(partials_dir="data/features_partials", batch_ids=[1, 2, 3]):
    """
    Carga solo algunos batches específicos.
    
    Args:
        partials_dir: Directorio con los parciales
        batch_ids: Lista de IDs de batches a cargar (ej: [1, 2, 3])
    
    Returns:
        DataFrame con solo esos batches concatenados
    """
    dfs = []
    
    for batch_id in batch_ids:
        # Buscar archivo que coincida con el batch_id
        pattern = str(Path(partials_dir) / f"partial_{batch_id:04d}_*.parquet")
        files = glob.glob(pattern)
        
        if not files:
            # Intentar con csv.gz
            pattern = str(Path(partials_dir) / f"partial_{batch_id:04d}_*.csv.gz")
            files = glob.glob(pattern)
        
        if files:
            f = files[0]  # Tomar el primero si hay varios con el mismo ID
            print(f"[LOAD] batch {batch_id}: {Path(f).name}", end=" ")
            
            if f.endswith(".parquet"):
                df = pd.read_parquet(f)
            else:
                df = pd.read_csv(f, compression='gzip')
            
            print(f"- {len(df)} filas")
            dfs.append(df)
        else:
            print(f"[WARNING] No se encontró archivo para batch {batch_id}")
    
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        print(f"\n[✓] Subset cargado: {len(combined)} filas de {len(batch_ids)} batches")
        return combined
    else:
        print(f"[ERROR] No se pudo cargar ningún batch")
        return None

# ============================================================================
# EJEMPLO 4: Cargar un solo archivo parcial
# ============================================================================
def load_single_partial(partials_dir="data/features_partials", batch_id=1):
    """Carga un único archivo parcial por su batch_id"""
    pattern = str(Path(partials_dir) / f"partial_{batch_id:04d}_*.parquet")
    files = glob.glob(pattern)
    
    if not files:
        pattern = str(Path(partials_dir) / f"partial_{batch_id:04d}_*.csv.gz")
        files = glob.glob(pattern)
    
    if files:
        f = files[0]
        print(f"[LOAD] {Path(f).name}")
        
        if f.endswith(".parquet"):
            df = pd.read_parquet(f)
        else:
            df = pd.read_csv(f, compression='gzip')
        
        print(f"[✓] Cargado: {len(df)} filas, {len(df.columns)} columnas")
        return df
    else:
        print(f"[ERROR] No se encontró archivo para batch {batch_id}")
        return None

# ============================================================================
# EJEMPLO DE USO
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("EJEMPLOS DE CARGA DE ARCHIVOS PARCIALES")
    print("=" * 70)
    
    partials_dir = "data/features_partials"
    
    # Opción 1: Cargar archivo combinado completo
    print("\n--- Opción 1: Archivo combinado completo ---")
    df_full = load_combined_dataframe(partials_dir)
    
    # Opción 2: Combinar todos los parciales manualmente
    print("\n--- Opción 2: Combinar todos los parciales ---")
    df_all = combine_all_partials(partials_dir)
    
    # Opción 3: Cargar solo algunos batches
    print("\n--- Opción 3: Cargar subset de batches [1, 2, 3] ---")
    df_subset = load_partial_subset(partials_dir, batch_ids=[1, 2, 3])
    
    # Opción 4: Cargar un solo batch
    print("\n--- Opción 4: Cargar solo batch 1 ---")
    df_single = load_single_partial(partials_dir, batch_id=1)
    
    print("\n" + "=" * 70)
    
    # Ejemplo de análisis rápido si se cargó algo
    if df_subset is not None:
        print("\nEJEMPLO DE ANÁLISIS DEL SUBSET:")
        print(f"Shape: {df_subset.shape}")
        print(f"\nPrimeras 3 filas:")
        print(df_subset.head(3))
        print(f"\nColumnas: {list(df_subset.columns[:15])}")
        print(f"\nDocumentos únicos: {df_subset['id_original'].nunique()}")
        print(f"Modelos: {df_subset['model'].unique()}")
