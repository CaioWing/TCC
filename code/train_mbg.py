"""Primeiro treino do detector no MBG-V2 (recortes nativos 640, divisao por regiao).

Pesos iniciais: YOLO11n COCO (models/yolo11n.pt). NAO usar os pesos do piloto: eles
foram treinados nas regioes J e L, e L agora e teste.
Parada antecipada e escolha do best.pt pela validacao (regioes F, I); o teste (E, H, L)
so e avaliado depois, uma vez, com evaluate().

Rodar no kernel do env 'tcc-venv' (DataLoader com workers falha em processos do PowerShell).
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('USERNAME', 'casti')

DATA = r'E:\TCC\data\mbg_v2\yolo_640\mbg_v2_640.yaml'
WEIGHTS = r'C:\Users\casti\projects\TCC\models\yolo11n.pt'
PROJECT = r'C:\Users\casti\TCC\runs'
NAME = 'mbg_y11n_640_v1'

CFG = dict(data=DATA, epochs=30, imgsz=640, batch=32, workers=4, device=0,
           project=PROJECT, name=NAME, exist_ok=True, seed=0, deterministic=False,
           patience=10, flipud=0.5,          # camera nadir: cima/baixo nao tem sentido fisico
           cache=False, plots=True, verbose=False)


def train(resume=False):
    from ultralytics import YOLO
    last = os.path.join(PROJECT, NAME, 'weights', 'last.pt')
    if resume and os.path.exists(last):
        return YOLO(last).train(resume=True)
    return YOLO(WEIGHTS).train(**CFG)


def evaluate(split='test', weights=None, conf=0.001):
    from ultralytics import YOLO
    w = weights or os.path.join(PROJECT, NAME, 'weights', 'best.pt')
    return YOLO(w).val(data=DATA, split=split, imgsz=640, batch=32, workers=4, device=0, conf=conf,
                       project=PROJECT, name=f'{NAME}_{split}', exist_ok=True, plots=True, verbose=True)
