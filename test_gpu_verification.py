"""
Script para verificar si Stylometrix está usando GPU correctamente
"""
import sys
sys.path.insert(0, 'dataframe')

from processer import check_gpu_availability, extract_features_from_dataset_batch
import pandas as pd

def main():
    print("\n" + "="*70)
    print("VERIFICACIÓN DE GPU PARA STYLOMETRIX")
    print("="*70 + "\n")
    
    # 1. Verificar disponibilidad de GPU
    print("1. Verificando disponibilidad de GPU...")
    check_gpu_availability()
    
    # 2. Crear un pequeño dataset de prueba
    print("\n2. Creando dataset de prueba...")
    test_data = {
        'id': [1, 2],
        'generation': [
            "This is a simple test sentence. It should be processed quickly.",
            "Another test document. With multiple sentences for processing."
        ],
        'model': ['test_model', 'test_model'],
        'domain': ['test', 'test']
    }
    df_test = pd.DataFrame(test_data)
    print(f"   Dataset creado con {len(df_test)} documentos")
    
    # 3. Ejecutar extracción de features
    print("\n3. Ejecutando extracción de features con GPU...\n")
    try:
        result = extract_features_from_dataset_batch(
            df_test, 
            sample_size=None, 
            use_gpu=True, 
            batch_size=32
        )
        print(f"\n[✓] Procesamiento completado exitosamente!")
        print(f"[✓] Resultados: {len(result)} filas con {len(result.columns)} características")
        print(f"\nPrimeras columnas: {list(result.columns[:10])}")
        
    except Exception as e:
        print(f"\n[✗] Error durante el procesamiento: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("VERIFICACIÓN COMPLETADA")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
