# probar otros modelos de bert
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
# A Symphony of Rage
# 1. Cargar CSV
df = pd.read_csv("diabetes_emociones.csv", encoding="utf-8")[["label", "text"]]

# print(df["label"].value_counts())
# df["label"] = df["label"].str.replace("negación_incredulidad", "negacion_incredulidad")

df["label"] = df["label"].map({    
                "miedo_ansiedad": 0,
                "tristeza": 1,
                "enojo_frustracion": 2,
                "culpa_verguenza": 3,
                "negacion_incredulidad": 4,
                "agotamiento_desesperanza": 5,
                "soledad_aislamiento": 6,
                "confusion_incertidumbre": 7,
                "preocupacion_familiar": 8,
                "resignacion": 9,
                "gratitud": 10,
                "esperanza_motivacion": 11,
                "orgullo": 12,
                "alivio": 13,
                "entusiasmo": 14,
                "alegria": 15})
df["label"] = df["label"].astype('int64')  



# 2. Dividir en train/test
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# 3. Convertir a Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 4. Tokenizador
model_name = "dccuchile/bert-base-spanish-wwm-cased" #La principal ventaja de este modelo de IA es su rendimiento. Requiere menos recursos computacionales
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 5. Definir mapeo de etiquetas
id2label = {0: "miedo_ansiedad", 1: "tristeza", 2: "enojo_frustracion", 
            3: "culpa_verguenza", 4: "negacion_incredulidad", 5: "agotamiento_desesperanza", 
            6: "soledad_aislamiento", 7: "confusion_incertidumbre", 8: "preocupacion_familiar", 
            9: "resignacion", 10: "gratitud", 11: "esperanza_motivacion", 12: "orgullo", 
            13: "alivio", 14: "entusiasmo", 15: "alegria"}
label2id = {"miedo_ansiedad": 0, "tristeza": 1, "enojo_frustracion": 2, 
            "culpa_verguenza": 3, "negacion_incredulidad": 4, "agotamiento_desesperanza": 5, 
            "soledad_aislamiento": 6, "confusion_incertidumbre": 7, "preocupacion_familiar": 8, 
            "resignacion": 9, "gratitud": 10, "esperanza_motivacion": 11, "orgullo": 12, 
            "alivio": 13, "entusiasmo": 14, "alegria": 15}

# 6. Cargar modelo con etiquetas personalizadas
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=16,
    id2label=id2label,
    label2id=label2id,
)

# 7. Configuración de entrenamiento (usa evaluation_strategy)
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,  # Guarda el modelo en la época con menor validation loss
    metric_for_best_model="eval_loss",
    learning_rate=3e-5, #aprende más o menos despacio para evitar sobreajuste
    per_device_train_batch_size=4,  #Este parámetro puede generar problemas de memoria insuficiente
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    weight_decay=0.1, #penaliza los pesos grandes para evitar sobreajuste
    logging_dir="./logs",
    logging_steps=50,
)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 8. Entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator  
)

# 9. Entrenar
trainer.train()

# 10. Evaluar
metrics = trainer.evaluate()
print("Resultados de evaluación:", metrics)

# 11. Guardar modelo y tokenizer
trainer.save_model("./emociones_diabetes_classifier")
tokenizer.save_pretrained("./emociones_diabetes_classifier")
