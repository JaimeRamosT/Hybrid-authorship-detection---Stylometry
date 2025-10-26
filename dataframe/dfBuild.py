# Instalar torch

import stylo_metrix as sm
import pandas as pd
from raid import run_detection, run_evaluation
from raid.utils import load_data

from processer import split_text_into_sentences

# Download the RAID dataset without adversarial attacks
or_train_noadv_df = load_data(split="train", include_adversarial=False)

print(f"Original training data shape: {or_train_noadv_df.shape}")

intCols = ['id','model', 'domain', 'title', 'prompt', 'generation']

# Copia del dataframe con columnas específicas
train_noadv_df = or_train_noadv_df.copy()
train_noadv_df = train_noadv_df[intCols]

filtered_by_domain = train_noadv_df[
    (train_noadv_df['domain'] != 'recipes')
    ]

# Separar documentos humanos y de IA
human_docs = filtered_by_domain[filtered_by_domain['model'] == 'human']
ai_docs = filtered_by_domain[filtered_by_domain['model'] != 'human']

# Calcular 50% del total deseado
total_samples = 1000
samples_per_class = total_samples // 2  # 500 de cada uno

# Samplear 500 humanos y 500 IA
human_sample = human_docs[['id', 'model', 'domain', 'generation']].sample(n=samples_per_class, random_state=50)
ai_sample = ai_docs[['id', 'model', 'domain', 'generation']].sample(n=samples_per_class, random_state=50)

# Combinar ambos samples
generation_sample = pd.concat([human_sample, ai_sample], ignore_index=True)

# Mezclar aleatoriamente
generation_sample = generation_sample.sample(frac=1, random_state=50).reset_index(drop=True)

print(f"Total muestras: {len(generation_sample)}")
print(f"Distribución por clase:")
print(generation_sample['model'].value_counts())
print(f"\nHumanos: {(generation_sample['model'] == 'human').sum()} ({(generation_sample['model'] == 'human').sum() / len(generation_sample) * 100:.1f}%)")
print(f"IA: {(generation_sample['model'] != 'human').sum()} ({(generation_sample['model'] != 'human').sum() / len(generation_sample) * 100:.1f}%)")


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

features_df = extract_features_from_dataset(generation_sample)


# Ordenar DF final
train_df = features_df.copy()

trainCols = ['id_encoded', 'sentence_num', 'model', 'domain', 'POS_VERB', 'POS_NOUN', 'POS_ADJ', 'POS_ADV', 'POS_DET', 'POS_INTJ', 'POS_CONJ', 'POS_PART', 'POS_NUM', 'POS_PREP', 'POS_PRO', 'L_REF', 'L_HASHTAG', 'L_MENTION', 'L_RT', 'L_LINKS', 'L_CONT_A', 'L_FUNC_A', 'L_CONT_T', 'L_FUNC_T', 'L_PLURAL_NOUNS', 'L_SINGULAR_NOUNS', 'L_PROPER_NAME', 'L_PERSONAL_NAME', 'L_NOUN_PHRASES', 'L_PUNCT', 'L_PUNCT_DOT', 'L_PUNCT_COM', 'L_PUNCT_SEMC', 'L_PUNCT_COL', 'L_PUNCT_DASH', 'L_POSSESSIVES', 'L_ADJ_POSITIVE', 'L_ADJ_COMPARATIVE', 'L_ADJ_SUPERLATIVE', 'L_ADV_POSITIVE', 'L_ADV_COMPARATIVE', 'L_ADV_SUPERLATIVE', 'PS_CONTRADICTION', 'PS_AGREEMENT', 'PS_EXAMPLES', 'PS_CONSEQUENCE', 'PS_CAUSE', 'PS_LOCATION', 'PS_TIME', 'PS_CONDITION', 'PS_MANNER', 'SY_QUESTION', 'SY_NARRATIVE', 'SY_NEGATIVE_QUESTIONS', 'SY_SPECIAL_QUESTIONS', 'SY_TAG_QUESTIONS', 'SY_GENERAL_QUESTIONS', 'SY_EXCLAMATION', 'SY_IMPERATIVE', 'SY_SUBORD_SENT', 'SY_SUBORD_SENT_PUNCT', 'SY_COORD_SENT', 'SY_COORD_SENT_PUNCT', 'SY_SIMPLE_SENT', 'SY_INVERSE_PATTERNS', 'SY_SIMILE', 'SY_FRONTING', 'SY_IRRITATION', 'SY_INTENSIFIER', 'SY_QUOT', 'VT_PRESENT_SIMPLE', 'VT_PRESENT_PROGRESSIVE', 'VT_PRESENT_PERFECT', 'VT_PRESENT_PERFECT_PROGR', 'VT_PRESENT_SIMPLE_PASSIVE', 'VT_PRESENT_PROGR_PASSIVE', 'VT_PRESENT_PERFECT_PASSIVE', 'VT_PAST_SIMPLE', 'VT_PAST_SIMPLE_BE', 'VT_PAST_PROGR', 'VT_PAST_PERFECT', 'VT_PAST_PERFECT_PROGR', 'VT_PAST_SIMPLE_PASSIVE', 'VT_PAST_POGR_PASSIVE', 'VT_PAST_PERFECT_PASSIVE', 'VT_FUTURE_SIMPLE', 'VT_FUTURE_PROGRESSIVE', 'VT_FUTURE_PERFECT', 'VT_FUTURE_PERFECT_PROGR', 'VT_FUTURE_SIMPLE_PASSIVE', 'VT_FUTURE_PROGR_PASSIVE', 'VT_FUTURE_PERFECT_PASSIVE', 'VT_WOULD', 'VT_WOULD_PASSIVE', 'VT_WOULD_PROGRESSIVE', 'VT_WOULD_PERFECT', 'VT_WOULD_PERFECT_PASSIVE', 'VT_SHOULD', 'VT_SHOULD_PASSIVE', 'VT_SHALL', 'VT_SHALL_PASSIVE', 'VT_SHOULD_PROGRESSIVE', 'VT_SHOULD_PERFECT', 'VT_SHOULD_PERFECT_PASSIVE', 'VT_MUST', 'VT_MUST_PASSIVE', 'VT_MUST_PROGRESSIVE', 'VT_MUST_PERFECT', 'VT_MST_PERFECT_PASSIVE', 'VT_CAN', 'VT_CAN_PASSIVE', 'VT_COULD', 'VT_COULD_PASSIVE', 'VT_CAN_PROGRESSIVE', 'VT_COULD_PROGRESSIVE', 'VT_COULD_PERFECT', 'VT_COULD_PERFECT_PASSIVE', 'VT_MAY', 'VT_MAY_PASSIVE', 'VT_MIGHT', 'VT_MIGHT_PASSIVE', 'VT_MAY_PROGRESSIVE', 'VT_MIGTH_PERFECT', 'VT_MIGHT_PERFECT_PASSIVE', 'VT_MAY_PERFECT_PASSIVE', 'ST_TYPE_TOKEN_RATIO_LEMMAS', 'ST_HERDAN_TTR', 'ST_MASS_TTR', 'ST_SENT_WRDSPERSENT', 'ST_SENT_DIFFERENCE', 'ST_REPETITIONS_WORDS', 'ST_REPETITIONS_SENT', 'ST_SENT_D_VP', 'ST_SENT_D_NP', 'ST_SENT_D_PP', 'ST_SENT_D_ADJP', 'ST_SENT_D_ADVP', 'L_I_PRON', 'L_HE_PRON', 'L_SHE_PRON', 'L_IT_PRON', 'L_YOU_PRON', 'L_WE_PRON', 'L_THEY_PRON', 'L_ME_PRON', 'L_YOU_OBJ_PRON', 'L_HIM_PRON', 'L_HER_OBJECT_PRON', 'L_IT_OBJECT_PRON', 'L_US_PRON', 'L_THEM_PRON', 'L_MY_PRON', 'L_YOUR_PRON', 'L_HIS_PRON', 'L_HER_PRON', 'L_ITS_PRON', 'L_OUR_PRON', 'L_THEIR_PRON', 'L_YOURS_PRON', 'L_THEIRS_PRON', 'L_HERS_PRON', 'L_OURS_PRON', 'L_MYSELF_PRON', 'L_YOURSELF_PRON', 'L_HIMSELF_PRON', 'L_HERSELF_PRON', 'L_ITSELF_PRON', 'L_OURSELVES_PRON', 'L_YOURSELVES_PRON', 'L_THEMSELVES_PRON', 'L_FIRST_PERSON_SING_PRON', 'L_SECOND_PERSON_PRON', 'L_THIRD_PERSON_SING_PRON', 'L_THIRD_PERSON_PLURAL_PRON', 'VF_INFINITIVE', 'G_PASSIVE', 'G_ACTIVE', 'G_PRESENT', 'G_PAST', 'G_FUTURE', 'G_MODALS_SIMPLE', 'G_MODALS_CONT', 'G_MODALS_PERFECT', 'AN', 'DDP', 'SVP', 'CDS', 'DDF', 'IS', 'PS', 'RE', 'ASF', 'ASM', 'OM', 'RCI', 'DMC', 'OR', 'QAS', 'PA', 'PR']

train_df = train_df[trainCols]

train_df = train_df.rename(columns={
    'id_encoded': 'id',
})

train_df = train_df.sort_values(by=['id', 'sentence_num']).reset_index(drop=True)

# Guardar el DataFrame final a un archivo CSV
pdtrainDF = pd.DataFrame(train_df)
pdtrainDF.to_csv('train_df.csv', index=False)