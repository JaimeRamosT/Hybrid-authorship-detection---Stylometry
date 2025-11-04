import stylo_metrix as sm
import re
import pandas as pd
import spacy

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
    # Limpiar saltos de línea múltiples y espacios extra
    textContent = re.sub(r'\n+', ' ', textContent)
    textContent = re.sub(r'\s+', ' ', textContent).strip()

    # Dividir en oraciones usando expresión regular
    # Detecta puntos, signos de exclamación e interrogación seguidos de espacio y mayúscula
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, textContent)

    # Limpiar oraciones vacías
    sentences = [s.strip() for s in sentences if s.strip()]

    # Agregar las oraciones a la lista de salida
    # outputList.extend(sentences)

    # print(f"[OK] Total de oraciones: {len(sentences)}")

    return sentences

def extract_features_from_dataset(df_original, sample_size=None):
    """
    Extrae features estilométricos a nivel de oración.

    Returns:
        DataFrame con estructura: id_original, model, domain, sentence_num, text, features...
    """
    if sample_size:
        df_original = df_original.sample(n=sample_size, random_state=42)

    # Inicializar StyloMetrix (sin guardar archivos)
    stylo = sm.StyloMetrix('en', debug=False)  # debug=False para evitar archivos

    all_results = []

    for idx, row in df_original.iterrows():
        # Dividir en oraciones (en memoria)
        sentences = split_text_into_sentences(row['generation'])

        # Extraer features para todas las oraciones del documento
        features_df = stylo.transform(sentences)

        # Agregar metadatos del documento original
        features_df.insert(0, 'id_original', row['id'])
        features_df.insert(1, 'model', row['model'])
        features_df.insert(2, 'domain', row['domain'])
        features_df.insert(3, 'sentence_num', range(len(sentences)))
        # La columna 'text' ya existe en features_df (viene de stylo.transform)

        all_results.append(features_df)

    # Concatenar todos los resultados
    final_df = pd.concat(all_results, ignore_index=True)

    return final_df

def extract_features_from_dataset_batch(df_original, sample_size=None, use_gpu=True, batch_size=32):
    """
    Versión optimizada con procesamiento en batch para GPU.
    """
    # Verificar disponibilidad de GPU
    print("=" * 60)
    print("DIAGNÓSTICO DE GPU")
    print("=" * 60)
    gpu_available = check_gpu_availability()
    
    if sample_size:
        df_original = df_original.sample(n=sample_size, random_state=42)

    # Intentar usar GPU si está disponible
    if use_gpu and gpu_available:
        try:
            spacy.prefer_gpu()
            print("[INFO] GPU activada para spaCy")
        except Exception as e:
            print(f"[WARNING] No se pudo activar GPU: {e}")
    elif use_gpu and not gpu_available:
        print("[WARNING] GPU solicitada pero no disponible - usando CPU")
    
    # Cargar modelo transformer explícitamente
    print("\nCargando modelo spaCy transformer...")
    nlp = spacy.load("en_core_web_trf")
    
    # Verificar si spaCy está usando GPU
    if spacy.prefer_gpu():
        print(f"[✓] spaCy usando GPU (dispositivo: cuda)")
    else:
        print(f"[!] spaCy usando CPU")
    
    # Verificar el dispositivo del modelo transformer
    if hasattr(nlp, 'get_pipe'):
        try:
            transformer = nlp.get_pipe('transformer')
            if hasattr(transformer, 'model') and hasattr(transformer.model, 'shims'):
                device = transformer.model.shims[0].device if transformer.model.shims else 'cpu'
                print(f"[INFO] Transformer en dispositivo: {device}")
        except Exception as e:
            print(f"[INFO] No se pudo verificar dispositivo del transformer: {e}")
    
    print("=" * 60)

    # Inicializar StyloMetrix con el modelo
    stylo = sm.StyloMetrix('en', debug=False, nlp=nlp)

    all_results = []
    total_docs = len(df_original)

    # Preparar todos los textos y metadatos
    all_sentences_data = []
    for idx, row in df_original.iterrows():
        sentences = split_text_into_sentences(row['generation'])
        for sent_num, sent in enumerate(sentences):
            all_sentences_data.append({
                'text': sent,
                'id_original': row['id'],
                'model': row['model'],
                'domain': row['domain'],
                'sentence_num': sent_num
            })

    # Procesar en batch
    texts = [item['text'] for item in all_sentences_data]

    print(f"Procesando {len(texts)} oraciones en batches de {batch_size}...")

    # Procesar todas las oraciones de una vez (aprovecha GPU al máximo)
    features_df = stylo.transform(texts)

    # Agregar metadatos
    for i, item in enumerate(all_sentences_data):
        features_df.loc[i, 'id_original'] = item['id_original']
        features_df.loc[i, 'model'] = item['model']
        features_df.loc[i, 'domain'] = item['domain']
        features_df.loc[i, 'sentence_num'] = item['sentence_num']

    # Reordenar columnas
    cols = ['id_original', 'model', 'domain', 'sentence_num'] + [c for c in features_df.columns if c not in ['id_original', 'model', 'domain', 'sentence_num']]
    features_df = features_df[cols]

    print(f"Extracción completada: {len(features_df)} oraciones")
    return features_df
