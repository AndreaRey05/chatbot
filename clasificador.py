from transformers import pipeline

clf = pipeline("text-classification", model="./emociones_diabetes_classifier", tokenizer="./emociones_diabetes_classifier")

#positive ejemplo
print(clf("me siento muy triste y agotado ultimamente y no sé que hacer para sentirme mejor, realemente no creo que pueda mejoar y me siento mal por mi familia que me cuida"))  

#negative ejemplo
#print(clf("Terrible film, waste of time and money. Poor acting."))