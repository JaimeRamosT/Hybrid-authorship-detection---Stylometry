"""
Script para combinar archivos parciales generados por dfBuild_chunked.py
"""
import sys
import os
from pathlib import Path

# Importar la función del script principal
from dfBuild_chunked import combine_partials

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Combina archivos parciales de características estilométricas"
    )
    parser.add_argument("--partials-dir", required=True, 
                       help="Directorio con los archivos partial_*.parquet o partial_*.csv.gz")
    parser.add_argument("--output", default=None,
                       help="Ruta del archivo combinado (default: <partials-dir>/features_combined.parquet)")
    args = parser.parse_args()
    
    # Determinar ruta de salida
    if args.output is None:
        args.output = os.path.join(args.partials_dir, "features_combined.parquet")
    
    # Validar directorio
    if not os.path.exists(args.partials_dir):
        print(f"[ERROR] Directorio no encontrado: {args.partials_dir}")
        exit(1)
    
    # Combinar
    try:
        df_combined = combine_partials(out_dir=args.partials_dir, output_path=args.output)
        print(f"\n[SUCCESS] Dataframe combinado exitosamente!")
        print(f"[INFO] Shape: {df_combined.shape}")
        print(f"[INFO] Columnas: {list(df_combined.columns[:10])}...")
    except Exception as e:
        print(f"[ERROR] No se pudo combinar: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

# Ejemplo de uso:
# python combine_partials.py --partials-dir "data/features_partials"
# python combine_partials.py --partials-dir "data/features_partials" --output "mi_dataframe_completo.parquet"
