import asyncio
import cv2
import base64
import gymnasium as gym
import ale_py.roms
from fastapi import FastAPI, WebSocket
from stable_baselines3 import DQN

# 1. Inicializamos el servidor web
app = FastAPI()

# 2. Cargamos el modelo y el entorno (fuera del bucle para que cargue solo una vez)
print("Cargando modelo...")
modelo = DQN.load("ddqn_mspacman_500k_pasos")
# Usamos rgb_array para poder capturar los fotogramas sin abrir una ventana de sistema
env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
print("Modelo cargado y entorno listo.")


# 3. Creamos el "túnel" de comunicación (WebSocket)
@app.websocket("/ws/play")
async def play_game(websocket: WebSocket):
    # Aceptamos la conexión del navegador del usuario
    await websocket.accept()
    obs, info = env.reset()

    try:
        # Bucle infinito del juego
        while True:
            # La IA decide el movimiento (sin exploración aleatoria)
            accion, _ = modelo.predict(obs, deterministic=True)

            # Ejecutamos el movimiento en el emulador
            obs, recompensa, done, truncado, info = env.step(accion)

            # Capturamos la pantalla actual
            frame = env.render()

            # OpenCV usa BGR en lugar de RGB, invertimos los colores para que no se vea azul
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Comprimimos la imagen a JPEG para que viaje rápido por internet
            _, buffer = cv2.imencode('.jpg', frame_bgr)

            # La convertimos a texto (Base64) para poder enviarla por el WebSocket
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            # Enviamos el fotograma al Frontend
            await websocket.send_text(frame_b64)

            # Si Pac-Man muere o se acaba el tiempo, reiniciamos la partida automáticamente
            if done or truncado:
                obs, info = env.reset()

            # Agregamos una pequeñísima pausa (aprox 30 FPS) para no saturar el navegador
            await asyncio.sleep(0.03)

    except Exception as e:
        # Si el usuario cierra la pestaña, terminamos la conexión limpiamente
        print("El cliente se ha desconectado.")