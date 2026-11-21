# Avaliação pareada do detector treinado na MBG-V2

Execute os comandos na raiz do TCC. O protocolo usa dois trechos de quatro segundos por voo, a um e dois terços do vídeo, e a mesma grade de recortes para todos os ambientes. A calibração geográfica vem do voo 3 do treino. As regiões F e I definem os parâmetros da associação. As regiões E, H e L fornecem o teste. O diretório `tmp/avaliacao_mbg` guarda exportações, máscaras, caches e previsões intermediárias. Os resultados numéricos ficam em `doc/dados/qualidade_memoria`.

1. Execute `python code/avaliacao_mbg/dados.py` para reproduzir as seleções desde o início dos voos e guardar as máscaras dos trechos.
2. Execute `python code/avaliacao_mbg/exportar.py` para exportar o checkpoint `runs/mbg_y11n_640_v1/weights/best.pt` para TorchScript e ONNX. O arquivo `origem_modelo.json` registra a época de maior mAP de validação e os hashes do checkpoint, da cópia de exportação e dos dois formatos exportados. Os motores TensorRT são construídos no equipamento de execução. INT8 usa 512 recortes de treino escolhidos por hash do nome, calibração por entropia e FP16 permitido.
3. Execute `python code/avaliacao_mbg/inferir.py torchscript_fp32 --split val` e `python code/avaliacao_mbg/avaliar.py calibrar`. O arquivo `protocolo.json` registra o hash dos pesos, os limiares e a separação de regiões.
4. Para cada uma das nove configurações enumeradas em `motores.py`, execute `python code/avaliacao_mbg/inferir.py MOTOR` e depois `python code/avaliacao_mbg/avaliar.py teste --motor MOTOR`. O lote é unitário. Um arquivo de previsões só fica disponível para avaliação após a conclusão de cada voo.
5. Execute `python code/avaliacao_mbg/consolidar.py`. Cada linha de `qualidade_teste.csv` cruza um motor e uma estratégia. As colunas `delta_*_pp` comparam com todos os recortes e todos os quadros do mesmo trecho no mesmo motor. `recall_trilha_iou` mede os objetos com ao menos uma caixa correta na imagem. As métricas de foco usam associação no solo. As AP usam os quadros completos e incluem oportunidades omitidas. Os arquivos por classe, trecho e região preservam as contagens.

6. Execute `python code/avaliacao_mbg/oportunidades_trilhas.py` para verificar se cada uma das trilhas comparadas tinha pelo menos 40% de uma caixa em algum recorte selecionado durante sua aparição completa no voo. O arquivo registra separadamente a oportunidade dentro dos quatro segundos e a existência de uma oportunidade em qualquer instante anotado. A etapa só consulta as anotações e a geometria da seleção.
7. Execute `python code/avaliacao_mbg/comparar_confiancas.py` para comparar FP16 e INT8 no processamento integral. Para cada trilha, o CSV registra o maior escore entre as caixas da classe correta pareadas com IoU de pelo menos 0,5. Uma célula vazia indica ausência de correspondência acima do piso de 0,001 usado na extração.

A execução local usa PyTorch CUDA, TorchScript, OpenCV, SciPy, NumPy, pandas, Ultralytics, ONNX Runtime GPU e TensorRT. No Windows, esta avaliação usa ONNX Runtime GPU 1.22.0 em `tmp/avaliacao_mbg/ortgpu` e os bindings e DLLs do TensorRT CUDA 12 versão 10.9.0.34 em `tmp/avaliacao_mbg/trt10`. As previsões locais servem à comparação de qualidade. Os tempos e as medições elétricas da Jetson pertencem ao ensaio de bancada separado.

## Mais de uma vista por célula

O experimento divide a imagem em uma a cinco faixas horizontais e registra, por célula do terreno, em quais faixas ela já foi analisada. Execute os comandos a partir da raiz após a avaliação original. A reprodução com uma faixa verifica quadro a quadro a coincidência com a seleção geográfica original. Cada ambiente verifica também as contagens, o F1 e as AP da comparação original com uma faixa.

1. Execute `python code/avaliacao_mbg/multivistas.py` para reconstruir as seleções completas dos seis voos, a cobertura geográfica, os custos por voo e as máscaras nos doze trechos.
2. Execute `python code/avaliacao_mbg/avaliar_multivistas.py trt_fp32 trt_fp16 trt_int8 ort_trt_fp32 ort_trt_fp16 ort_cuda_fp32 ort_cpu_fp32 torchscript_fp32 torchscript_fp16 consolidar` para aplicar essas máscaras às previsões do modelo treinado e consolidar os resultados.
3. Execute `python code/avaliacao_mbg/controle_orcamento.py` para a sensibilidade ao quadro inicial com orçamento determinado pelos próprios trechos em TensorRT FP16.
4. Execute `python code/avaliacao_mbg/figura_multivistas.py` para reproduzir a figura de qualidade e demanda. Os CSV públicos ficam em `doc/dados/qualidade_memoria/multivistas`. As taxas, a cobertura e os picos usam os voos completos. As contagens de detecções e alarmes usam os doze trechos.

### Seguimento nos 18 trechos adicionais

O plano está em `doc/dados/qualidade_memoria/seguimento/protocolo.json`. Na raiz do projeto, execute em sequência:

```
python code/avaliacao_mbg/seguimento.py preparar
python code/avaliacao_mbg/seguimento.py inferir
python code/avaliacao_mbg/seguimento.py avaliar
python code/avaliacao_mbg/figura_seguimento.py
```

A primeira etapa verifica a reprodução das máscaras originais e a separação entre os quadros. A segunda usa o mesmo motor TensorRT FP16 e calcula uma única previsão por recorte solicitado. A terceira usa os limiares originais e confere o pareamento dos orçamentos por trecho.

Na coluna `orcamento`, `natural` preserva as propostas da memória e `inteiros` arredonda a demanda daquela meta para baixo em grupos de 24. Em `fixo_bruto` todas as metas recebem a menor demanda do trecho. Em `fixo` esse mínimo é arredondado para quadros inteiros. A fase -1 identifica a memória, cuja seleção independe do relógio. Cada fase temporal é uma alternativa com o mesmo orçamento.
