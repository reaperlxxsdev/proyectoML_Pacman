import asyncio
import cv2
import base64
import os
import traceback
import gdown
import ale_py.roms
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from stable_baselines3 import DQN

# --- NUEVOS IMPORTS PARA ATARI ---
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

file_id = '1KeJDGDuFoovIOK-qXYY3vjGW-78CKMGP'
url = f'https://drive.google.com/uc?id={file_id}'
output_file = 'ddqn_mspacman_500k_pasos.zip'

if not os.path.exists(output_file):
    print("Descargando el cerebro de la IA desde Google Drive...")
    gdown.download(url, output_file, quiet=False)

app = FastAPI()

print("Cargando modelo...")
modelo = DQN.load(output_file)

# --- LA SOLUCIÓN MÁGICA: Entorno Vectorizado ---
# Preparamos el entorno exactamente con los filtros con los que fue entrenado
env = make_atari_env("ALE/MsPacman-v5", n_envs=1, env_kwargs={"render_mode": "rgb_array"})
env = VecFrameStack(env, n_stack=4)

print("Modelo cargado y entorno listo. Esperando conexión...")


@app.websocket("/ws/play")
async def play_game(websocket: WebSocket):
    await websocket.accept()

    # VecEnv de SB3 solo devuelve 'obs', no 'info'
    obs = env.reset()

    try:
        while True:
            accion, _ = modelo.predict(obs, deterministic=True)

            # VecEnv.step devuelve 4 valores en lugar de 5 (agrupa done y truncado)
            obs, recompensa, done, info = env.step(accion)

            # El render sigue extrayendo la matriz (210, 160, 3) para nosotros
            frame = env.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', frame_bgr)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            await websocket.send_text(frame_b64)

            # En VecEnv, 'done' es un arreglo booleano (ej. [False])
            if done[0]:
                obs = env.reset()

            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        print("El cliente cerró la pestaña (Desconexión normal).")

    except Exception as e:
        print(f"🔥 ALERTA ROJA: {e}")
        traceback.print_exc()