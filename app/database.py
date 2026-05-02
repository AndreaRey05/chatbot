import os
from supabase import create_client, Client
from dotenv import load_dotenv
from pathlib import Path
import random

load_dotenv(Path(__file__).parent.parent / ".env")

# ─── Conexión ─────────────────────────────────────────────────────────────────
def get_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("Faltan las variables SUPABASE_URL o SUPABASE_KEY en el .env")

    return create_client(url, key)


# ─── Tablas disponibles (una por emoción) ─────────────────────────────────────
TABLAS_EMOCIONES = [
    "enojo_frustracion",
    "miedo_ansiedad",
    "tristeza",
    "negacion_incredulidad",
    "resignacion",
    "culpa_verguenza",
    "preocupacion_familiar",
    "agotamiento_desesperanza",
    "confusion_incertidumbre",
    "soledad_aislamiento",
    "gratitud",
    "esperanza_motivacion",
    "orgullo",
    "entusiasmo",
    "alegria",
    "alivio"
    # Agrega aquí el resto de tus 16 tablas con el nombre exacto de Supabase
]


# ─── Consulta de contenido por emoción ────────────────────────────────────────
def obtener_contenido(etiqueta: str) -> list[dict]:
    """
    Recibe la etiqueta que devuelve BERT (ej: "enojo_frustracion")
    y consulta la tabla correspondiente en Supabase.
    Retorna lista de recursos o lista vacía si no encuentra nada.
    """
    # Verificar que la etiqueta corresponde a una tabla existente
    if etiqueta not in TABLAS_EMOCIONES:
        print(f"[WARN] Etiqueta '{etiqueta}' no tiene tabla asociada.")
        return []

    try:
        supabase = get_client()
        
        respuesta = (
            supabase
            .table(etiqueta)
            .select("id, titulo, tipo_recurso, enlace_recurso")
            .execute()
        )
        datos = respuesta.data or []
        return [random.choice(datos)] if datos else []

    except Exception as e:
        print(f"[ERROR] No se pudo consultar la tabla '{etiqueta}': {e}")
        return []


# ─── Prueba local ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    etiqueta_prueba = "agotamiento_desesperanza"
    resultados = obtener_contenido(etiqueta_prueba)

    if resultados:
        print(f"\nContenido encontrado para '{etiqueta_prueba}':")
        for item in resultados:
            print(f"  - [{item['tipo_recurso']}] {item['titulo']}")
            print(f"    {item['enlace_recurso']}")
    else:
        print(f"No se encontró contenido para '{etiqueta_prueba}'.")