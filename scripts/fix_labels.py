import pandas as pd

df = pd.read_csv("diabetes_emociones.csv", encoding="utf-8")
print(df["label"].value_counts())

# Ver exactamente qué etiquetas hay
'''print("Etiquetas únicas:")
print(df["label"].unique())

# Ver los caracteres de la etiqueta problemática
mask = df["label"].str.contains("negaci", na=False)
print("\nVersiones de negaci encontradas:")
print(df[mask]["label"].unique())

# Corregir
df["label"] = df["label"].str.strip()
df["label"] = df["label"].str.replace("negación_incredulidad", "negacion_incredulidad", regex=False)
'''
# Verificar
#print("\nDespués de corregir:")

'''
# Guardar
df.to_csv("diabetes_emociones.csv", encoding="utf-8", index=False)
print("\nArchivo guardado correctamente")
'''