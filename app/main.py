from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chat_diabetes import chat_diabetes, detectar_crisis, clasificar_emocion
from database import obtener_contenido, guardar_mensaje, obtener_historial, obtener_sesiones_usuario

app = FastAPI(title="Chatbot Diabetes API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sesiones: dict[str, list] = {}

class MensajeRequest(BaseModel):
    sesion_id: str
    mensaje: str
    usuario_id: str = None

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

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: MensajeRequest):
    if not request.mensaje.strip():
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

    if request.sesion_id not in sesiones:
        historial_db = obtener_historial(request.sesion_id)
        sesiones[request.sesion_id] = [
            {"role": h["role"], "content": h["contenido"]}
            for h in historial_db
        ]

    historial = sesiones[request.sesion_id]
    tipo_crisis = detectar_crisis(request.mensaje)
    es_crisis = tipo_crisis is not None

    respuesta, historial_actualizado, contenido = chat_diabetes(
        request.mensaje, historial
    )
    print(f"[DEBUG] Contenido desde chat_diabetes: {contenido}")

    sesiones[request.sesion_id] = historial_actualizado

    emocion = ""
    if not es_crisis:
        emocion, _ = clasificar_emocion(request.mensaje)

    guardar_mensaje(
        sesion_id=request.sesion_id,
        role="user",
        contenido=request.mensaje,
        usuario_id=request.usuario_id,
        emocion=emocion,
        es_crisis=es_crisis,
    )
    guardar_mensaje(
        sesion_id=request.sesion_id,
        role="assistant",
        contenido=respuesta,
        usuario_id=request.usuario_id,
    )

    return ChatResponse(
        reply=respuesta,
        emocion=emocion,
        contenido=[ContenidoItem(**item) for item in contenido],
        es_crisis=es_crisis,
    )

@app.delete("/chat/{sesion_id}")
async def reset_sesion(sesion_id: str):
    if sesion_id in sesiones:
        del sesiones[sesion_id]
    return {"status": "ok", "mensaje": "Sesión reiniciada."}

@app.get("/historial/{sesion_id}")
async def get_historial(sesion_id: str):
    return obtener_historial(sesion_id)

@app.get("/sesiones/{usuario_id}")
async def get_sesiones(usuario_id: str):
    return obtener_sesiones_usuario(usuario_id)