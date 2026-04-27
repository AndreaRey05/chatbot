import ollama
import time

# ─── Configuración ────────────────────────────────────────────────────────────
MAX_HISTORIAL = 10  # Máximo de mensajes a conservar en contexto

SISTEMA = """Eres un asistente empático especializado en apoyar a pacientes con diabetes en México,
de entre 45 y 60 años. Tu objetivo es escuchar, validar sus emociones y brindar apoyo emocional y técnicas
de regulación cuando el paciente lo necesite.

Reglas importantes:
- Responde siempre en español mexicano coloquial y cálido
- No des consejos médicos específicos
- Valida lo que siente el paciente antes de cualquier otra cosa
- Cuando el paciente exprese ansiedad, pánico, angustia o estrés, ofrece una técnica de regulación concreta
  como respiración profunda, técnica 5-4-3-2-1, respiración diafragmática o grounding
- Explica la técnica de manera simple y paso a paso
- Sé breve y claro, máximo 5 oraciones
- No uses lenguaje clínico ni tecnicismos

Técnicas que puedes usar:
- Respiración 4-7-8: inhala 4 segundos, retén 7, exhala 8
- Técnica 5-4-3-2-1: nombra 5 cosas que ves, 4 que escuchas, 3 que puedes tocar, 2 que hueles, 1 que saboreas
- Respiración diafragmática: respira lento y profundo inflando el abdomen
- Grounding: apoya los pies en el piso y siente el contacto con la tierra"""

# ─── Palabras clave de crisis ─────────────────────────────────────────────────
CRISIS_SUICIDIO = [
    "quiero morir", "no quiero vivir", "mejor muerto", "quitarme la vida",
    "suicidarme", "suicidio", "matarme", "ya no quiero estar aquí",
    "para qué seguir", "no tiene caso seguir viviendo", "desaparecer para siempre",
    # Variaciones comunes en adultos mayores
    "ya no quiero seguir", "me quiero quitar la vida", "no quiero seguir viviendo",
    "quiero quitarme la vida", "ganas de morir", "ya no quiero nada",
    "para que vivir", "para qué vivir", "mejor ya no despertar",
]

CRISIS_AUTOLESION = [
    "hacerme daño", "lastimarme", "cortarme", "golpearme",
    "autolesión", "quiero lastimarme", "hacerme algo",
    # Variaciones
    "quiero hacerme daño", "me voy a lastimar", "hacerme algo malo",
]

CRISIS_PANICO = [
    "ataque de pánico", "no puedo respirar", "me voy a morir",
    "el corazón se me sale", "me estoy ahogando", "no puedo más",
    "siento que me muero", "me está dando algo", "ataque de ansiedad",
    # Variaciones comunes
    "me falta el aire", "siento el corazón muy rápido", "me está dando un ataque",
    "no me puedo calmar", "siento que me va a dar algo",
]

# ─── Detección de crisis ──────────────────────────────────────────────────────
def detectar_crisis(mensaje: str) -> str | None:
    """
    Detecta si el mensaje contiene señales de crisis.
    Normaliza el texto para tolerar errores ortográficos menores.
    Retorna el tipo de crisis o None.
    """
    # Normalización básica: minúsculas y sin acentos para mayor tolerancia
    mensaje_lower = mensaje.lower()
    mensaje_norm = (mensaje_lower
                    .replace("á", "a").replace("é", "e")
                    .replace("í", "i").replace("ó", "o")
                    .replace("ú", "u").replace("ü", "u"))

    def contiene(frases: list[str]) -> bool:
        for frase in frases:
            frase_norm = (frase
                          .replace("á", "a").replace("é", "e")
                          .replace("í", "i").replace("ó", "o")
                          .replace("ú", "u"))
            if frase_norm in mensaje_norm:
                return True
        return False

    if contiene(CRISIS_SUICIDIO):
        return "suicidio"
    if contiene(CRISIS_AUTOLESION):
        return "autolesion"
    if contiene(CRISIS_PANICO):
        return "panico"

    return None

# ─── Respuestas de crisis ─────────────────────────────────────────────────────
def respuesta_crisis(tipo_crisis: str) -> str:
    respuestas = {
        "suicidio": (
            "Escucho que estás pasando por algo muy difícil y me importa mucho cómo te sientes.\n"
            "No estás solo/a en esto.\n\n"
            "Por favor comunícate ahora con la Línea de la Vida: 800 911 2000, "
            "es gratuita, confidencial y disponible las 24 horas.\n\n"
            "¿Hay alguien de confianza cerca de ti ahorita con quien puedas estar?"
        ),
        "autolesion": (
            "Lo que me cuentas me dice que estás cargando algo muy pesado "
            "y quiero que sepas que estoy aquí contigo.\n\n"
            "Por favor llama ahora a la Línea de la Vida: 800 911 2000, "
            "tienen personas capacitadas para ayudarte en este momento.\n\n"
            "¿Puedes alejarte de cualquier objeto con el que pudieras hacerte daño mientras hablamos?"
        ),
        "panico": (
            "Estoy aquí contigo, vamos a pasar esto juntos.\n\n"
            "Haz esto ahorita: pon los dos pies bien apoyados en el piso, "
            "inhala lentamente contando hasta 4, aguanta 2 segundos y exhala contando hasta 6. "
            "Repítelo tres veces.\n\n"
            "El ataque de pánico no te va a hacer daño, va a pasar. "
            "¿Puedes intentar la respiración conmigo?"
        ),
    }
    return respuestas.get(tipo_crisis, "")

# ─── Chat principal ───────────────────────────────────────────────────────────
def chat_diabetes(mensaje_usuario: str, historial: list) -> tuple[str, list]:
    """
    Procesa un mensaje del usuario y retorna (respuesta, historial_actualizado).
    Recibe y retorna el historial para que cada sesión sea independiente.
    """
    # 1. Detectar crisis antes que cualquier otra cosa
    tipo_crisis = detectar_crisis(mensaje_usuario)
    if tipo_crisis:
        return respuesta_crisis(tipo_crisis), historial

    # 2. Agregar mensaje al historial
    historial.append({
        "role": "user",
        "content": mensaje_usuario,
    })

    # 3. Limitar historial para no saturar el contexto del modelo
    if len(historial) > MAX_HISTORIAL:
        historial = historial[-MAX_HISTORIAL:]

    # 4. Llamar a Ollama con manejo de errores
    try:
        respuesta = ollama.chat(
            model="llama3",
            messages=[{"role": "system", "content": SISTEMA}] + historial,
        )
        contenido = respuesta["message"]["content"]
    except ollama.ResponseError as e:
        contenido = f"Hubo un problema con el modelo: {e}. Verifica que Ollama esté corriendo."
    except Exception:
        contenido = "Hubo un problema de conexión. Por favor intenta de nuevo."

    # 5. Agregar respuesta al historial
    historial.append({
        "role": "assistant",
        "content": contenido,
    })

    return contenido, historial

# ─── Ejecución local (prueba) ─────────────────────────────────────────────────
if __name__ == "__main__":
    historial_sesion = []  # Historial por sesión, no global

    print("Chatbot: Hola, estoy aquí para escucharte. ¿Cómo te has sentido hoy?")

    while True:
        mensaje = input("\nTú: ").strip()

        if not mensaje:
            continue
        if mensaje.lower() == "salir":
            print("Chatbot: Cuídate mucho. Aquí estaré cuando me necesites.")
            break

        inicio = time.time()  # ← mide cada respuesta individualmente

        respuesta, historial_sesion = chat_diabetes(mensaje, historial_sesion)

        fin = time.time()

        print(f"\nChatbot: {respuesta}")
        print(f"\n⏱️ Tiempo de respuesta: {fin - inicio:.3f} segundos")
        print("-" * 50)