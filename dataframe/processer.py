import re

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
    
    print(f"[OK] Total de oraciones: {len(sentences)}")

    return sentences
