from transformers import pipeline
from pathlib import Path

# Ruta absoluta al modelo sin importar desde dónde lo corras
MODEL_PATH = Path(__file__).parent.parent / "model" / "emociones_diabetes_classifier"

clf = pipeline("text-classification", model=str(MODEL_PATH), tokenizer=str(MODEL_PATH))

print(clf("odio demasiado cuandp me siento asi, le echo ganas y aún así nada resulta bien"))