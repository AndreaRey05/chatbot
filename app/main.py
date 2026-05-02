from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chat_diabetes import chat_diabetes, detectar_crisis, clasificar_emocion

app = FastAPI(title="Chatbot Diabetes API")

# Permite peticiones desde Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Almacén de sesiones en memoria ──────────────────────────────────────────
# Guarda el historial de cada usuario por separado
# { "sesion_id": [lista de mensajes] }
sesiones: dict[str, list] = {}

# ─── Modelos de datos ─────────────────────────────────────────────────────────
class MensajeRequest(BaseModel):
    sesion_id: str      # identificador único del usuario (Flutter lo genera)
    mensaje: str        # lo que escribió el usuario

class ContenidoItem(BaseModel):
    id: int
    titulo: str
    tipo_recurso: str
    enlace_recurso: str

class ChatResponse(BaseModel):
    reply: str
    emocion: str
    contenido: list[ContenidoItem]
    es_crisis: bool

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Verifica que el servidor esté corriendo."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: MensajeRequest):
    """
    Recibe un mensaje del usuario y devuelve:
    - reply: respuesta empática de Ollama
    - emocion: etiqueta clasificada por BERT
    - contenido: recurso sugerido de Supabase
    - es_crisis: si se detectó una situación de crisis
    """
    if not request.mensaje.strip():
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

    # Recuperar o crear historial de la sesión
    if request.sesion_id not in sesiones:
        sesiones[request.sesion_id] = []

    historial = sesiones[request.sesion_id]

    # Detectar crisis antes de cualquier otra cosa
    tipo_crisis = detectar_crisis(request.mensaje)
    es_crisis = tipo_crisis is not None

    # Llamar al chatbot
    respuesta, historial_actualizado, contenido = chat_diabetes(
        request.mensaje, historial
    )

    # Guardar historial actualizado
    sesiones[request.sesion_id] = historial_actualizado

    # Obtener emoción clasificada (solo si no es crisis)
    emocion = ""
    if not es_crisis:
        emocion, _ = clasificar_emocion(request.mensaje)

    return ChatResponse(
        reply=respuesta,
        emocion=emocion,
        contenido=[ContenidoItem(**item) for item in contenido],
        es_crisis=es_crisis,
    )


@app.delete("/chat/{sesion_id}")
async def reset_sesion(sesion_id: str):
    """Limpia el historial de una sesión."""
    if sesion_id in sesiones:
        del sesiones[sesion_id]
    return {"status": "ok", "mensaje": "Sesión reiniciada."}