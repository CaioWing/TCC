"""Runners de lote unitário, compartilhando pesos, entrada e pós-processamento."""
import hashlib
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from filelock import FileLock

from dados import CACHE, DATA, OUT

ENGINES = ("torchscript_fp32", "torchscript_fp16", "ort_cpu_fp32", "ort_cuda_fp32",
           "trt_fp32", "trt_fp16", "trt_int8", "ort_trt_fp32", "ort_trt_fp16")
DLL_HANDLES = []
TRT_OPT = 1


def setup():
    sys.path[:0] = [str(CACHE / "ortgpu"), str(CACHE / "trt10")]
    if os.name == "nt":
        for path in [Path(torch.__file__).parent / "lib", CACHE / "trt10/tensorrt_libs"]:
            if path.exists():
                os.environ["PATH"] = str(path) + os.pathsep + os.environ["PATH"]
                DLL_HANDLES.append(os.add_dll_directory(str(path)))
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def blob(image):
    return cv2.dnn.blobFromImage(image, 1/255, swapRB=True)


def build_engine(key):
    import tensorrt_bindings as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(CACHE / "detector.onnx")):
        raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
    config = builder.create_builder_config()
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.builder_optimization_level = TRT_OPT
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.clear_flag(trt.BuilderFlag.TF32)
    if key != "trt_fp32":
        config.set_flag(trt.BuilderFlag.FP16)
    if key == "trt_int8":
        names = sorted((DATA / "yolo_640/images/train").glob("*.jpg"),
                       key=lambda p: hashlib.sha256(p.name.encode()).digest())[:512]
        assert len(names) == 512
        (CACHE / "calibracao_int8.json").write_text(json.dumps([p.name for p in names]))

        class Calibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self):
                super().__init__()
                self.images = iter(names)
                self.tensor = torch.empty((1, 3, 640, 640), device="cuda")

            def get_batch_size(self):
                return 1

            def get_batch(self, _):
                name = next(self.images, None)
                if name is None:
                    return None
                self.tensor.copy_(torch.from_numpy(blob(cv2.imread(str(name)))))
                torch.cuda.synchronize()
                return [self.tensor.data_ptr()]

            def read_calibration_cache(self):
                return None

            def write_calibration_cache(self, data):
                (CACHE / "calibracao_int8.bin").write_bytes(bytes(data))

        config.set_flag(trt.BuilderFlag.INT8)
        calibrator = Calibrator()
        config.int8_calibrator = calibrator
    data = builder.build_serialized_network(network, config)
    if data is None:
        raise RuntimeError("TensorRT não construiu " + key)
    path = CACHE / f"{key}.opt{TRT_OPT}.engine"
    tmp = path.with_suffix(".partial")
    tmp.write_bytes(bytes(data))
    tmp.replace(path)


def runner(key, *, verificar_provedor=False):
    if key.startswith("torchscript"):
        mod = torch.jit.load(str(CACHE / "detector.torchscript"), map_location="cuda").eval()
        dtype = torch.float16 if key.endswith("fp16") else torch.float32
        mod.to(dtype=dtype)
        tensor = torch.zeros((1,3,640,640),device="cuda",dtype=dtype)
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(5):
                mod(tensor)
        torch.cuda.current_stream().wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = mod(tensor)
        if isinstance(output,(tuple,list)):
            output=output[0]
        # Confere se o grafo usa o conteúdo atual da entrada, ainda em lote unitário.
        tensor.fill_(.23)
        reference = mod(tensor)
        if isinstance(reference,(tuple,list)):
            reference=reference[0]
        graph.replay()
        torch.testing.assert_close(output,reference,rtol=0,atol=0)

        def run_ts(x):
            tensor.copy_(torch.from_numpy(x))
            graph.replay()
            return output.float().cpu().numpy()
        run_ts.resources = mod
        return run_ts
    if key.startswith("ort_"):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 3
        opts.inter_op_num_threads = 1
        if verificar_provedor:
            opts.enable_profiling = True
            opts.profile_file_prefix = str(CACHE / ("perfil_" + key))
        cuda = ("CUDAExecutionProvider", {"use_tf32": 0})
        providers = ["CPUExecutionProvider"]
        if key.startswith("ort_cuda"):
            providers.insert(0, cuda)
        elif key.startswith("ort_trt"):
            cache = CACHE / f"{key}_opt{TRT_OPT}"
            cache.mkdir(exist_ok=True)
            providers = [("TensorrtExecutionProvider", {"trt_fp16_enable": key.endswith("fp16"),
                          "trt_engine_cache_enable": True, "trt_engine_cache_path": str(cache),
                          "trt_builder_optimization_level": TRT_OPT, "trt_max_workspace_size": 1 << 30}),
                         cuda, "CPUExecutionProvider"]
        compilation = key.startswith("ort_trt")
        with FileLock(str(CACHE / "compilacao_trt.lock")) if compilation else nullcontext():
            session = ort.InferenceSession(str(CACHE / "detector.onnx"), opts, providers=providers)
            if compilation:
                session.run(None, {session.get_inputs()[0].name: np.zeros((1,3,640,640),np.float32)})
        expected = providers[0] if isinstance(providers[0], str) else providers[0][0]
        if session.get_providers()[0] != expected:
            raise RuntimeError(f"{key}: provedor solicitado {expected} inativo: {session.get_providers()}")
        inp = session.get_inputs()[0].name
        if verificar_provedor:
            session.run(None, {inp: np.zeros((1,3,640,640), np.float32)})
            trace = json.loads(Path(session.end_profiling()).read_text(encoding="utf-8"))
            nodes = {}
            for event in trace:
                provider = event.get("args", {}).get("provider")
                if provider:
                    nodes[provider] = nodes.get(provider, 0) + 1
            if expected not in nodes:
                raise RuntimeError(f"{key}: nenhum nó foi executado por {expected}: {nodes}")
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / ("provedor_" + key + ".json")).write_text(json.dumps(nodes, indent=2))
        return lambda x: session.run(None, {inp: x})[0]
    import tensorrt_bindings as trt
    path = CACHE / f"{key}.opt{TRT_OPT}.engine"
    if not path.exists():
        with FileLock(str(CACHE / "compilacao_trt.lock")):
            if not path.exists():
                build_engine(key)
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    layers = json.loads(engine.create_engine_inspector().get_engine_information(
        trt.LayerInformationFormat.JSON))
    io_types = set()
    for layer in layers.get("Layers", []):
        for tensor in layer.get("Inputs", []) + layer.get("Outputs", []):
            fmt = tensor.get("Format/Datatype", tensor.get("DataType", ""))
            if fmt:
                io_types.add(fmt)
    audit = dict(engine_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 otimizacao=TRT_OPT, camadas=engine.num_layers, formatos=sorted(io_types))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"motor_{key}.json").write_text(json.dumps(audit, indent=2))
    ctx = engine.create_execution_context()
    inputs, outputs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        tensor = torch.empty(tuple(engine.get_tensor_shape(name)), device="cuda",
                             dtype=torch.from_numpy(np.empty(0, trt.nptype(engine.get_tensor_dtype(name)))).dtype)
        ctx.set_tensor_address(name, tensor.data_ptr())
        (inputs if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs).append(tensor)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    def run_trt(x):
        with torch.cuda.stream(stream):
            inputs[0].copy_(torch.from_numpy(x))
            if not ctx.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("Falha na inferência TensorRT.")
            return outputs[0].float().cpu().numpy()
    # No TensorRT os recursos nativos devem sobreviver ao contexto.
    run_trt.resources = (runtime, engine)
    return run_trt


def postprocess(pred):
    p = pred[0]
    if p.shape != (16, 8400):
        raise ValueError(f"Esperados os 12 escores do modelo MBG, obtido {p.shape}")
    scores = p[4:].max(axis=0)
    anchors = np.flatnonzero(scores >= .001)
    labels = p[4:, anchors].argmax(axis=0)
    evaluated = labels < 6
    anchors, labels = anchors[evaluated], labels[evaluated]
    scores = scores[anchors]
    b = p[:4, anchors].T.copy()
    b[:, :2] -= b[:, 2:] / 2
    inds = np.asarray(cv2.dnn.NMSBoxesBatched(b.tolist(), scores.tolist(), labels.tolist(),
                                               .001, .7)).astype(int).ravel()[:300]
    b = b[inds]
    b[:, 2:] += b[:, :2]
    return np.column_stack((np.clip(b, 0, 640), scores[inds], labels[inds])).astype(np.float32)
