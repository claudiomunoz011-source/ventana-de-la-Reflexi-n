import os
import time
import json
import math
from flask import Flask, request, jsonify, session, render_template
from dotenv import load_dotenv
import numpy as np

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.urandom(24)

# Configuración de Google GenAI (Gemini)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

client = None
if GEMINI_API_KEY:
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Advertencia: No se pudo inicializar el cliente de Google GenAI: {e}")

PAUSE_SECONDS = 60
MAX_PROMPTS = 3

QUESTIONS = [
    "¿Es el tiempo una ilusión de la conciencia o una propiedad real del universo?",
    "¿El libre albedrío existe realmente o es una ilusión determinada por leyes físicas?",
    "¿Puede una máquina llegar a tener conciencia o es la subjetividad un privilegio biológico?",
    "Si el bien y el mal son constructos humanos, ¿existe una moral objetiva?",
    "¿Es el lenguaje una herramienta para describir la realidad, o la realidad está construida por el lenguaje?"
]

def obtener_embedding(texto):
    if not client or not texto:
        return None
    try:
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=texto[:8000]
        )
        if hasattr(response, 'embedding') and hasattr(response.embedding, 'values'):
            return response.embedding.values
        elif hasattr(response, 'embeddings') and len(response.embeddings) > 0:
            return response.embeddings[0].values
        return None
    except Exception as e:
        print(f"Error al obtener embedding con Gemini: {e}")
        return None

def distancia_coseno(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0.0
    vec1 = np.array(emb1)
    vec2 = np.array(emb2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generar_pregunta', methods=['POST', 'GET'])
def generar_pregunta():
    if not client:
        return jsonify({'error': 'La clave GEMINI_API_KEY no está configurada.'}), 500
    try:
        from google.genai import types
        sistema_preguntas = """
        Eres un filósofo educador especializado en formular dilemas éticos y preguntas filosóficas profundas.
        - Genera UNA SOLA pregunta o dilema ético/filosófico provocador en español.
        - Debe ser clara, intrigante y apta para debate escolar/universitario.
        - No agregues introducciones, numeración ni saludos. Entrega únicamente el texto de la pregunta.
        - Longitud máxima: 20 a 35 palabras.
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Genera un dilema o pregunta filosófica/ética profunda e inspiradora.",
            config=types.GenerateContentConfig(
                system_instruction=sistema_preguntas,
                temperature=0.95,
                max_output_tokens=150
            )
        )
        pregunta_generada = response.text.strip()
        return jsonify({'status': 'ok', 'pregunta': pregunta_generada})
    except Exception as e:
        return jsonify({'error': f'Error al generar pregunta con Gemini: {str(e)}'}), 500

@app.route('/api/iniciar', methods=['POST'])
def iniciar():
    data = request.json or {}
    pregunta_custom = data.get('pregunta', '').strip()
    pregunta_idx = data.get('pregunta_idx', 0)
    
    if pregunta_custom:
        pregunta_final = pregunta_custom
    else:
        pregunta_final = QUESTIONS[pregunta_idx % len(QUESTIONS)]

    session.clear()
    session['pregunta'] = pregunta_final
    session['opinion_inicial'] = data.get('opinion_inicial', '')
    session['prompts'] = []
    session['respuestas_ia'] = []
    session['tiempo_inicio_pausa'] = None
    session['reflexion_final'] = ''
    session['finalizado'] = False
    return jsonify({'status': 'ok', 'pregunta': session['pregunta']})

@app.route('/api/preguntar', methods=['POST'])
def preguntar():
    data = request.json or {}
    prompt_usuario = data.get('prompt', '').strip()
    if not prompt_usuario:
        return jsonify({'error': 'El prompt no puede estar vacío'}), 400
    if len(prompt_usuario) < 10:
        return jsonify({'error': 'Mínimo 10 caracteres.'}), 400
    if len(session.get('prompts', [])) >= MAX_PROMPTS:
        return jsonify({'error': 'Ya realizaste tus 3 preguntas.'}), 400

    if not client:
        return jsonify({'error': 'La clave GEMINI_API_KEY no está configurada en el servidor.'}), 500

    sistema = """
    Eres el "Oráculo de la Duda", un asistente filosófico.
    - Responde con claridad, profundidad y provocación.
    - Nunca des respuestas definitivas; siempre deja espacio para la duda.
    - Menciona corrientes filosóficas si es posible.
    - Tus respuestas deben tener entre 80 y 120 palabras.
    - NUNCA dejes oraciones a la mitad: asegúrate de terminar siempre tus frases y finalizar con una pregunta o reflexión filosófica completa.
    """
    try:
        from google.genai import types
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Pregunta del usuario sobre la premisa: '{session.get('pregunta', '')}'. Opinión inicial del usuario: '{session.get('opinion_inicial', '')}'. Pregunta específica actual del usuario: {prompt_usuario}",
            config=types.GenerateContentConfig(
                system_instruction=sistema,
                temperature=0.85,
                max_output_tokens=2000
            )
        )
        texto_respuesta = response.text.strip()
    except Exception as e:
        return jsonify({'error': f'Error al consultar Gemini (Google AI Studio): {str(e)}'}), 500

    session['prompts'].append(prompt_usuario)
    session['respuestas_ia'].append(texto_respuesta)
    session.modified = True

    profundidad = 0
    if len(session['prompts']) >= 2:
        emb1 = obtener_embedding(session['prompts'][-2])
        emb2 = obtener_embedding(session['prompts'][-1])
        if emb1 and emb2:
            sim = distancia_coseno(emb1, emb2)
            profundidad = max(0, 1 - sim) * 100

    return jsonify({
        'status': 'ok',
        'respuesta': texto_respuesta,
        'prompt_numero': len(session['prompts']),
        'profundidad': round(profundidad, 2),
        'max_prompts': MAX_PROMPTS
    })

@app.route('/api/iniciar_pausa', methods=['POST'])
def iniciar_pausa():
    session['tiempo_inicio_pausa'] = time.time()
    session.modified = True
    return jsonify({'status': 'ok', 'duracion': PAUSE_SECONDS})

@app.route('/api/verificar_pausa', methods=['GET'])
def verificar_pausa():
    inicio = session.get('tiempo_inicio_pausa')
    if inicio is None:
        return jsonify({'restante': 0, 'terminado': True})
    transcurrido = time.time() - inicio
    restante = max(0, PAUSE_SECONDS - transcurrido)
    return jsonify({'restante': int(restante), 'terminado': restante <= 0, 'total': PAUSE_SECONDS})

@app.route('/api/guardar_reflexion', methods=['POST'])
def guardar_reflexion():
    data = request.json or {}
    texto = data.get('texto', '').strip()
    if len(texto) < 20:
        return jsonify({'error': 'Mínimo 20 palabras.'}), 400
    session['reflexion_final'] = texto
    session.modified = True
    return jsonify({'status': 'ok'})

@app.route('/api/evaluar_contraste', methods=['GET'])
def evaluar_contraste():
    pregunta = session.get('pregunta', '')
    opinion_inicial = session.get('opinion_inicial', '')
    prompts = session.get('prompts', [])
    respuestas_ia = session.get('respuestas_ia', [])
    reflexion_humana = session.get('reflexion_final', '')

    texto_ia_completo = " ".join(respuestas_ia)

    emb_ia = obtener_embedding(texto_ia_completo)
    emb_humano = obtener_embedding(reflexion_humana)
    riesgo_dependencia = distancia_coseno(emb_ia, emb_humano) if emb_ia and emb_humano else 0.0

    palabras_ia = set(texto_ia_completo.lower().split())
    palabras_humano = set(reflexion_humana.lower().split())
    coincidencias = palabras_ia.intersection(palabras_humano)
    solo_ia = palabras_ia - palabras_humano
    solo_humano = palabras_humano - palabras_ia

    insignias = []
    if len(solo_humano) > 10:
        insignias.append("🏆 'El Espejo Roto': Aportaste muchas ideas propias")
    if len(session.get('prompts', [])) == MAX_PROMPTS:
        insignias.append("🧠 'Prometeo': Completaste las 3 preguntas")
    if riesgo_dependencia < 0.4:
        insignias.append("⚔️ 'Estoico': Tu reflexión es muy independiente de la IA.")
    elif riesgo_dependencia > 0.8:
        insignias.append("⚠️ Alerta: Tu texto es muy similar al de la IA.")

    return jsonify({
        'pregunta': pregunta,
        'opinion_inicial': opinion_inicial,
        'prompts': prompts,
        'respuestas_ia': respuestas_ia,
        'texto_ia_completo': texto_ia_completo,
        'reflexion_humana': reflexion_humana,
        'riesgo_dependencia': round(riesgo_dependencia * 100, 2),
        'coincidencias': list(coincidencias)[:20],
        'solo_ia': list(solo_ia)[:20],
        'solo_humano': list(solo_humano)[:20],
        'insignias': insignias
    })

@app.route('/api/finalizar', methods=['POST'])
def finalizar():
    session['finalizado'] = True
    session.modified = True
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

