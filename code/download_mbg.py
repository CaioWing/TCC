"""Baixa os vídeos do MBG-V2 (SMT/COPPE/UFRJ) e extrai o conteúdo enquanto baixa, sem guardar o .tgz.

Uso: python code/download_mbg.py [números dos vídeos]   (sem argumentos: os 26 vídeos)
Saída: data/mbg_v2/raw/flight-mbg-v2-XX-regionY/...  e o log em data/mbg_v2/download_log.txt

É retomável no nível do vídeo: um vídeo com o marcador _DONE é pulado. O servidor não aceita Range,
então um vídeo interrompido recomeça do início.
"""
import concurrent.futures as cf
import os
import sys
import tarfile
import time

import requests

SITE = "https://www02.smt.ufrj.br/~tvdigital/database/mosquito/"
BASE = SITE + "data/002/tgz/"
GEOJSON = SITE + "docs/page/02/areas_RJ_Brazil.geojson"
REG = "AABBCCDDEEFFGGHHIIJJKKLLMM"
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mbg_v2")
RAW = os.path.join(ROOT, "raw")
LOG = os.path.join(ROOT, "download_log.txt")
TRIES = 10


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


class Counter:
    def __init__(self, f):
        self.f, self.n = f, 0

    def read(self, k=-1):
        b = self.f.read(k if k and k > 0 else 1 << 20)
        self.n += len(b)
        return b


def get(num):
    for k in range(1, TRIES + 1):
        try:
            return _get(num)
        except Exception as e:
            log(f"video {num}: tentativa {k} falhou ({type(e).__name__}); recomeçando o vídeo")
            time.sleep(10 * k)
    raise RuntimeError(f"video {num}: desisti após {TRIES} tentativas")


def _get(num):
    name = f"flight-mbg-v2-{num:02d}-region{REG[num - 1]}"
    done = os.path.join(RAW, name, "_DONE")
    if os.path.exists(done):
        log(f"{name}: já baixado")
        return name, 0
    t0 = time.time()
    r = requests.get(BASE + name + ".tgz", stream=True, timeout=(30, 300))
    r.raise_for_status()
    r.raw.decode_content = True  # o servidor aplica gzip de transporte por cima do .tgz
    c = Counter(r.raw)
    last = 0
    raw_prefix = os.path.abspath(RAW) + os.sep
    with tarfile.open(fileobj=c, mode="r|gz") as tf:
        for m in tf:
            dest = os.path.join(RAW, m.name)
            if not os.path.abspath(dest).startswith(raw_prefix):
                continue
            if m.isdir():
                os.makedirs(dest, exist_ok=True)
                continue
            if not m.isfile():
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            src = tf.extractfile(m)
            with open(dest + ".part", "wb") as f:
                while True:
                    b = src.read(8 << 20)
                    if not b:
                        break
                    f.write(b)
                    if c.n - last > 1 << 30:
                        last = c.n
                        log(f"{name}: {c.n / 2**30:.1f} GiB, {c.n / 2**20 / (time.time() - t0):.0f} MiB/s")
            os.replace(dest + ".part", dest)
    open(done, "w").close()
    log(f"{name}: OK, {c.n / 2**30:.2f} GiB em {(time.time() - t0) / 60:.1f} min")
    return name, c.n


def get_geojson():
    dest = os.path.join(ROOT, "areas_RJ_Brazil.geojson")
    if os.path.exists(dest):
        return
    r = requests.get(GEOJSON, timeout=(30, 300))
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)
    log(f"geojson das regiões salvo ({len(r.content) / 2**10:.0f} KiB)")


def main(nums, workers=2):
    os.makedirs(RAW, exist_ok=True)
    failed = []
    try:
        get_geojson()
    except Exception as e:
        log(f"ERRO geojson: {type(e).__name__}: {e}")
        failed.append("geojson")
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(get, n): n for n in nums}
        for fut in cf.as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                log(f"ERRO: {type(e).__name__}: {e}")
                failed.append(futs[fut])
    log(f"FIM: {len(nums) - len([x for x in failed if x != 'geojson'])}/{len(nums)} vídeos completos; falhas: {failed or 'nenhuma'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main([int(x) for x in sys.argv[1:]] or list(range(1, 27))))
