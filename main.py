import asyncio
import cv2
import base64
import os
import traceback
import gdown
import gymnasium as gym
import ale_py.roms
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from stable_baselines3 import DQN

# --- SECCIÓN: DESCARGA AUTOMÁTICA DEL MODELO ---
file_id = '1KeJDGDuFoovIOK-qXYY3vjGW-78CKMGP'
url = f'https://drive.google.com/uc?id={file_id}'
output_file = 'ddqn_mspacman_500k_pasos.zip'

if not os.path.exists(output_file):
    print("Descargando el cerebro de la IA desde Google Drive...")
    gdown.download(url, output_file, quiet=False)
# -----------------------------------------------------

app = FastAPI()

print("Cargando modelo...")
modelo = DQN.load(output_file)
env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
print("Modelo cargado y entorno listo. Esperando conexión...")

@app.websocket("/ws/play")
async def play_game(websocket: WebSocket):
    # Aceptamos la conexión del navegador del usuario
    await websocket.accept()
    obs, info = env.reset()

    try:
        # Bucle infinito del juego
        while True:
            # La IA decide el movimiento
            accion, _ = modelo.predict(obs, deterministic=True)

            # Ejecutamos el movimiento en el emulador
            obs, recompensa, done, truncado, info = env.step(accion)

            # Capturamos la pantalla actual
            frame = env.render()

            # OpenCV usa BGR en lugar de RGB
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Comprimimos la imagen a JPEG
            _, buffer = cv2.imencode('.jpg', frame_bgr)

            # Convertimos a texto (Base64)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            # Enviamos el fotograma al Frontend
            await websocket.send_text(frame_b64)

            # Reinicio de partida
            if done or truncado:
                obs, info = env.reset()

            # Pausa de 30 FPS
            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        # Esto solo se activa si tú cierras la pestaña de Netlify
        print("El cliente cerró la pestaña (Desconexión normal).")
        
    except Exception as e:
        # Esto atrapará el error de la Red Neuronal e imprimirá qué le duele
        print(f"🔥 ALERTA ROJA: El juego colapsó con este error: {e}")
        traceback.print_exc()
