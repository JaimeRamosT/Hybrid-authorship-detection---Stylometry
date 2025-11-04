import os, gc, glob, time
import re
import pandas as pd
import spacy
import stylo_metrix as sm
from pathlib import Path
from datetime import datetime

def check_gpu_availability():
    """
    Verifica si CUDA/GPU está disponible para PyTorch y spaCy.
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
            print(f"[✓] CUDA disponible: {cuda_available}")
            print(f"[✓] Dispositivos GPU: {device_count}")
            print(f"[✓] GPU principal: {device_name}")
            return True
        else:
            print("[!] CUDA NO disponible - se usará CPU")
            return False
    except ImportError:
        print("[!] PyTorch no instalado - no se puede verificar GPU")
        return False

def split_text_into_sentences(textContent):
    textContent = re.sub(r'\n+', ' ', textContent)
    textContent = re.sub(r'\s+', ' ', textContent).strip()
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, textContent)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def batches_from_df(df, max_sentences_per_batch=5000):
    """Generador: itera documentos y acumula oraciones hasta alcanzar el límite"""
    batch = []
    batch_count = 0
    for idx, row in df.iterrows():
        sentences = split_text_into_sentences(row['generation'])
        for sent_num, sent in enumerate(sentences):
            batch.append({
                'text': sent,
                'id_original': row['id'],
                'model': row['model'],
                'domain': row['domain'],
                'sentence_num': sent_num
            })
            if len(batch) >= max_sentences_per_batch:
                batch_count += 1
                yield batch_count, batch
                batch = []
    if batch:
        batch_count += 1
        yield batch_count, batch

def save_df_partial(df, out_dir, batch_id):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname_base = f"partial_{batch_id:04d}_{ts}"
    # intenta parquet primero
    try:
        path = out_dir / f"{fname_base}.parquet"
        df.to_parquet(path, index=False)
        return str(path)
    except Exception as e:
        # fallback a csv gzip
        path = out_dir / f"{fname_base}.csv.gz"
        df.to_csv(path, index=False, compression='gzip')
        return str(path)

def process_and_save_in_batches(df_original, out_dir="features_partial", max_sentences_per_batch=5000, use_gpu=True):
    # Diagnóstico de GPU
    print("=" * 70)
    print("DIAGNÓSTICO DE GPU Y CONFIGURACIÓN")
    print("=" * 70)
    gpu_available = check_gpu_availability()
    
    # Preferir GPU si está disponible
    if use_gpu and gpu_available:
        try:
            spacy.prefer_gpu()
            print("[INFO] GPU activada para spaCy")
        except Exception as e:
            print(f"[WARNING] No se pudo activar GPU: {e}")
    elif use_gpu and not gpu_available:
        print("[WARNING] GPU solicitada pero no disponible - usando CPU")
    
    # Cargar modelo transformer
    print("\nCargando modelo spaCy transformer...")
    nlp = spacy.load("en_core_web_trf")
    
    # Verificar dispositivo de spaCy
    if spacy.prefer_gpu():
        print(f"[✓] spaCy usando GPU")
    else:
        print(f"[!] spaCy usando CPU")
    
    # Verificar dispositivo del transformer
    if hasattr(nlp, 'get_pipe'):
        try:
            transformer = nlp.get_pipe('transformer')
            if hasattr(transformer, 'model') and hasattr(transformer.model, 'shims'):
                device = transformer.model.shims[0].device if transformer.model.shims else 'cpu'
                print(f"[INFO] Transformer en dispositivo: {device}")
        except Exception as e:
            print(f"[INFO] No se pudo verificar dispositivo del transformer: {e}")
    
    print(f"[INFO] Tamaño de batch: {max_sentences_per_batch} oraciones")
    print("=" * 70 + "\n")
    
    stylo = sm.StyloMetrix('en', debug=False, nlp=nlp)

    # detectar parciales ya guardados para reanudar
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(glob.glob(str(out_dir / "partial_*.parquet")) + glob.glob(str(out_dir / "partial_*.csv.gz")))
    processed_batches = set()
    for p in existing:
        # extrae batch_id del nombre si sigue el patrón "partial_{id:04d}_"
        bn = Path(p).stem
        parts = bn.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            processed_batches.add(int(parts[1]))

    print(f"[INFO] Procesando en lotes. Parciales existentes (se omiten): {sorted(processed_batches)}")
    if processed_batches:
        print(f"[INFO] Se omitirán {len(processed_batches)} batches ya procesados")
    total_saved = 0
    start_time = time.time()

    for batch_id, batch in batches_from_df(df_original, max_sentences_per_batch=max_sentences_per_batch):
        if batch_id in processed_batches:
            print(f"[SKIP] batch {batch_id} ya procesado.")
            continue

        batch_start_time = time.time()
        texts = [b['text'] for b in batch]
        
        # transform devuelve DataFrame con una fila por texto
        print(f"\n[PROCESSING] batch {batch_id} - {len(texts)} oraciones...")
        features_df = stylo.transform(texts)

        # Agregar metadatos a features_df (más eficiente creando columnas directamente)
        ids = [b['id_original'] for b in batch]
        models = [b['model'] for b in batch]
        domains = [b['domain'] for b in batch]
        snums = [b['sentence_num'] for b in batch]

        features_df.insert(0, 'id_original', ids)
        features_df.insert(1, 'model', models)
        features_df.insert(2, 'domain', domains)
        features_df.insert(3, 'sentence_num', snums)

        saved_path = save_df_partial(features_df, out_dir, batch_id)
        batch_elapsed = time.time() - batch_start_time
        oraciones_por_segundo = len(features_df) / batch_elapsed if batch_elapsed > 0 else 0
        
        print(f"[SAVED] batch {batch_id} -> {saved_path}")
        print(f"        {len(features_df)} filas | {batch_elapsed:.1f}s | {oraciones_por_segundo:.2f} oraciones/s")
        total_saved += len(features_df)

        # liberar memoria
        del features_df
        gc.collect()
        time.sleep(0.2)  # pequeña pausa para liberar recursos si es necesario

    total_elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"[DONE] Procesado completo")
    print(f"       Filas totales guardadas: {total_saved}")
    print(f"       Tiempo total: {total_elapsed/60:.2f} minutos")
    print(f"       Velocidad promedio: {total_saved/total_elapsed:.2f} oraciones/s")
    print(f"{'=' * 70}")
    return str(out_dir)

def combine_partials(out_dir="features_partial", output_path="features_combined.parquet"):
    """Combina todos los archivos parciales en uno solo"""
    print(f"\n{'=' * 70}")
    print("COMBINANDO ARCHIVOS PARCIALES")
    print(f"{'=' * 70}")
    
    files = sorted(glob.glob(os.path.join(out_dir, "partial_*.parquet")) + 
                   glob.glob(os.path.join(out_dir, "partial_*.csv.gz")))
    
    if not files:
        raise FileNotFoundError("No se encontraron archivos parciales en " + str(out_dir))
    
    print(f"[INFO] Encontrados {len(files)} archivos parciales")
    
    dfs = []
    total_rows = 0
    
    for i, f in enumerate(files, 1):
        print(f"[LOAD] ({i}/{len(files)}) {Path(f).name}", end=" ")
        if f.endswith(".parquet"):
            df = pd.read_parquet(f)
        else:
            df = pd.read_csv(f, compression='gzip')
        
        print(f"- {len(df)} filas")
        total_rows += len(df)
        dfs.append(df)
    
    print(f"\n[INFO] Concatenando {total_rows} filas totales...")
    combined = pd.concat(dfs, ignore_index=True)
    
    # Intentar guardar en parquet, si falla, csv.gz
    print(f"[INFO] Guardando archivo combinado...")
    try:
        combined.to_parquet(output_path, index=False)
        file_size = os.path.getsize(output_path) / (1024*1024)  # MB
        print(f"[✓] SAVED combinado -> {output_path}")
        print(f"    {len(combined)} filas | {len(combined.columns)} columnas | {file_size:.2f} MB")
    except Exception as e:
        print(f"[WARNING] No se pudo guardar en Parquet: {e}")
        out_csv = Path(output_path).with_suffix(".csv.gz")
        combined.to_csv(out_csv, index=False, compression='gzip')
        file_size = os.path.getsize(out_csv) / (1024*1024)  # MB
        print(f"[✓] SAVED combinado -> {out_csv}")
        print(f"    {len(combined)} filas | {len(combined.columns)} columnas | {file_size:.2f} MB")
    
    print(f"{'=' * 70}\n")
    return combined

# ---- ejemplo de uso ----
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Extrae características estilométricas de textos en batches con soporte GPU"
    )
    parser.add_argument("--input", help="CSV/Parquet con las columnas id,generation,model,domain", required=True)
    parser.add_argument("--outdir", help="Directorio para parciales", default="features_partial")
    parser.add_argument("--chunksize", type=int, help="Número máximo de oraciones por parcial", default=10000)
    parser.add_argument("--combine", action="store_true", help="Combinar parciales después de procesar")
    parser.add_argument("--no-gpu", action="store_true", help="Desactivar uso de GPU")
    args = parser.parse_args()

    # Validar que el archivo de entrada existe
    if not os.path.exists(args.input):
        print(f"[ERROR] Archivo no encontrado: {args.input}")
        exit(1)

    # Cargar input
    print(f"\n[INFO] Cargando dataset desde: {args.input}")
    try:
        if args.input.endswith(".parquet"):
            df_in = pd.read_parquet(args.input)
        else:
            df_in = pd.read_csv(args.input)
        
        print(f"[✓] Dataset cargado: {len(df_in)} documentos")
        
        # Validar columnas requeridas
        required_cols = ['id', 'generation', 'model', 'domain']
        missing_cols = [col for col in required_cols if col not in df_in.columns]
        if missing_cols:
            print(f"[ERROR] Columnas faltantes en el dataset: {missing_cols}")
            print(f"[ERROR] Columnas requeridas: {required_cols}")
            print(f"[ERROR] Columnas encontradas: {list(df_in.columns)}")
            exit(1)
        
        print(f"[✓] Columnas validadas: {required_cols}")
        
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el archivo: {e}")
        exit(1)

    # Procesar
    use_gpu = not args.no_gpu
    process_and_save_in_batches(df_in, out_dir=args.outdir, 
                                max_sentences_per_batch=args.chunksize,
                                use_gpu=use_gpu)

    if args.combine:
        combine_partials(out_dir=args.outdir, 
                        output_path=os.path.join(args.outdir, "features_combined.parquet"))

# Ejemplo de ejecución:
# python dfBuild_chunked.py --input "or_train_df.csv" --outdir "data/features_partials" --chunksize 3000