# Comparação das estratégias de seleção na MBG-V2

## Resultados dos seis voos completos usados no TCC

A memória por quadrantes solicita um recorte na primeira vista de cada quadrante durante a passagem de um ponto do terreno pela imagem. O controle temporal distribui o mesmo total de recortes por voo. Nas tabelas de trabalho, filtre `metodo` por `quadrantes` e `intervalo_pareado`. Os identificadores `cobertura` e `intervalo` nessas tabelas pertencem ao ensaio anterior.

- `python code/avaliacao_mbg/quadrantes_voos.py` reproduz a comparação nos vídeos originais a 40 m e grava os resultados em `doc/dados/qualidade_memoria/quadrantes`.
- `python code/avaliacao_mbg/altitude_quadrantes.py` aplica as escalas de 50 a 80 m às mesmas seleções. O subdiretório `quadrantes/altitude` reúne as medidas por voo, por trilha e no conjunto. A referência integral a 40 m vem do ensaio completo em `altitude_integral`.

## Teste da origem da malha

`python code/avaliacao_mbg/quadrantes_primeiro_gps.py` ancora a malha na primeira posição válida de GPS de cada voo. O primeiro registro também define a latitude de referência para a conversão em metros. O controle temporal recebe a cota efetivamente usada pela nova malha, com a mesma fase do ensaio original. Os resultados a 40 m ficam em `doc/dados/qualidade_memoria/quadrantes/primeiro_gps`. A coluna `origem` distingue os dois ensaios. No intervalo, essa coluna identifica de qual malha veio a cota.

## Avaliações anteriores

| Estratégia | Identificador na avaliação inicial |
|---|---|
| Processamento integral | `todos` |
| A cada 30 quadros | `cada30` |
| Intervalo ajustado por voo | `pareada` |
| Seleção na entrada | `imediata` |
| Seleção na saída | `adiada` |

Nos arquivos de multivistas, `geo` identifica a seleção na saída e `tempo`, a amostragem com intervalo ajustado por voo. O arquivo `controle_fases_fp16.csv` registra o controle por trecho nos 12 trechos iniciais, com arredondamento para quadros inteiros. Nos arquivos de seguimento, `memoria` identifica a seleção na saída e `temporal` identifica o controle por trecho, com a mesma quantidade de recortes por trecho para as duas estratégias em cada comparação.

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

### Vistas por passagem e reunião de caixas na fronteira dos recortes

Com as previsões integrais já calculadas nos mesmos 24 trechos de teste e nos oito de validação, execute a partir da raiz do projeto.

```text
python code/avaliacao_mbg/passagens.py preparar
python code/avaliacao_mbg/passagens.py nms
python code/avaliacao_mbg/passagens.py fronteiras
python code/avaliacao_mbg/passagens.py origem
python code/avaliacao_mbg/auditar_fronteiras.py
python code/avaliacao_mbg/figura_passagem.py
python code/avaliacao_mbg/figura_vistas_da_passagem.py
python code/avaliacao_mbg/figura_uniao_fronteira.py
```

O diretório `doc/dados/qualidade_memoria/passagens` contém os eventos geográficos do voo completo, o exemplo da figura e os resultados da seleção e do controle temporal pareado. Cada referência é identificada por vídeo, trecho, classe e trilha. Os arquivos `pareamento_*` permitem conferir a interseção com os acertos do processamento integral sob o mesmo pós-processamento. A etapa `origem` repete a seleção com outras três origens da grade no terreno. Em `doc/dados/qualidade_memoria/fronteiras`, a auditoria registra as duas caixas de cada par e confronta cada uma com a parte visível da anotação no próprio recorte.

Para estender o teste a mais vistas, execute `python code/avaliacao_mbg/curva_passagens.py` e depois `python code/avaliacao_mbg/figura_curva_passagens.py`. A avaliação distribui K quadros pela distância percorrida na passagem completa e pontua as previsões nos mesmos 24 trechos. A última meta usa todos os quadros de cada passagem. Os resultados em `passagens/curva` incluem a contagem por trecho, estratégia e fase temporal, com verificação das máscaras e dos resultados originais para K de 1 a 3.

O ensaio de quadros inteiros nos eixos da câmera usa os mesmos 24 trechos e o mesmo pós-processamento. Execute, na raiz do projeto, os comandos abaixo. O primeiro grava resultados por trecho, fase e cota em `doc/dados/qualidade_memoria/passagens/camera` e verifica a reprodução do integral. O segundo calibra o filtro causal nos oito trechos de F/I com previsões TensorRT FP16 da cadência de 30 quadros. Zero no limiar reproduz o calendário original, contado a partir do início do vídeo.

```bash
python code/avaliacao_mbg/movimento_camera.py
python code/avaliacao_mbg/filtro_cadencia_camera.py
python code/avaliacao_mbg/figura_movimento_camera.py
```

As figuras com exemplos reais das decisões de seleção podem ser reproduzidas com os seguintes comandos na raiz do repositório.

```bash
python code/avaliacao_mbg/figura_entrada_saida_celula.py
python code/avaliacao_mbg/figura_exemplo_temporal.py
python code/avaliacao_mbg/figura_saida_faixas.py
```

O gráfico de qualidade nos três ambientes de execução é gerado a partir de `qualidade_teste.csv` com `python code/avaliacao_mbg/figura_qualidade_motores.py`.

## Ensaios exploratórios fora do escopo do TCC

Os experimentos a seguir ficam disponíveis para reprodução. A etapa de inferência da primeira experiência também fornece previsões integrais reutilizadas no ensaio por passagem.

### Segunda vista por pouco detalhe

Execute as etapas a partir da raiz do projeto.

```
python code/avaliacao_mbg/adaptativa.py selecionar
python code/avaliacao_mbg/adaptativa.py inferir
python code/avaliacao_mbg/adaptativa.py avaliar
```

A seleção calibra o detalhe local nos oito trechos de validação de F e I e percorre os seis voos de teste desde o início. A inferência calcula todos os recortes dos doze novos trechos em TensorRT FP16 com o modelo treinado. A avaliação reutiliza as previsões integrais dos doze trechos iniciais e as novas, confere a igualdade do orçamento por trecho nas quatro fases temporais e grava o protocolo, as contagens por trecho e voo, os totais e a tabela do ensaio em `doc/dados/qualidade_memoria/adaptativa`.

### Recortes por cobertura e Rodízio de recortes

Execute `python code/avaliacao_mbg/selecao_trajetoria.py` com as previsões integrais de validação em TorchScript FP32 e de teste em TensorRT FP16. O script escolhe a quantidade de quadros e o peso da segunda vista em F e I. Nos seis voos de teste, a pose da câmera e a grade de células determinam as máscaras de seleção. O rodízio usa os mesmos instantes e cotas da regra geográfica. O programa verifica se o controle temporal reproduz as contagens originais em cada trecho e fase e grava o protocolo, a validação, as contagens e a tabela do ensaio em `doc/dados/qualidade_memoria/trajetoria`.
