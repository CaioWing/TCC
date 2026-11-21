# Proposta de apresentação dos resultados da Jetson

Análise de 17/11/2026, com três agentes Terra: auditoria numérica, leitura dos capítulos e avaliação de protocolo/gráficos. Os capítulos, dados originais e PDF do TCC não foram alterados nesta revisão.

## Recomendação

Adicionar agora uma seção de **caracterização de desempenho e energia em bancada**, após o treinamento do detector, distinguindo-a da futura avaliação do detector MBG embarcado. O resultado central é o compromisso entre precisão numérica, latência, energia e qualidade da detecção. TensorRT FP16 é o candidato inicial mais bem sustentado para a próxima etapa: nesta rodada, reduziu energia e latência frente a FP32 com mAP muito próximo. INT8 oferece ganho adicional, acompanhado de perda de mAP que impede tratá-lo como vencedor sem uma restrição explícita de qualidade.

O ensaio usa YOLO11n oficial COCO-80, não os pesos MBG de 12 classes apresentados no capítulo. O benchmark fim a fim processa 95 quadros do carrinho, já decodificados, de 640 × 480 pixels; a rede recebe tensor 1 × 3 × 640 × 640. Isso não mede a decodificação do vídeo aéreo, geração/seleção de recortes nativos, memória geográfica nem associação de detecções. Portanto, fornece uma comparação de implementações, e apenas uma aproximação inicial para dimensionar a missão.

## O que foi conferido

- Os 21 pares configuração/modo tiveram os arquivos de latência NPY comparados com os JSONs: contagens, média, mediana, p90, p99, extremos, desvio e FPS coincidem dentro da precisão de armazenamento. Erro máximo observado de aproximadamente 0,00000532 ms nas estatísticas de latência.
- As médias de potência de `power.csv`, calculadas nas janelas registradas, reproduzem os CSVs; a energia total por inferência reproduz P/FPS. A média ponderada por integração trapezoidal difere da média simples em no máximo 0,0074%, no intervalo entre primeira e última amostra de cada janela.
- As janelas ociosas deram 4,76778 e 4,75726 W; a média é 4,76252 W. A energia incremental também foi conferida. A soma pré-processamento + inferência + pós-processamento fica de 0,008 a 0,026 ms abaixo da latência total, diferença pequena compatível com os trechos de código externos aos cronômetros das etapas.
- Avaliação COCO: 500 nomes únicos de imagens; calibração: 512 nomes únicos, sem interseção. Os nove dumps de avaliação têm os mesmos 500 nomes; os nove dumps do carrinho têm os mesmos 95 nomes. Esta verificação usa nomes, não hashes do conteúdo das imagens.
- O mAP abaixo é o resultado da avaliação fornecida. O protocolo e os conjuntos foram inspecionados; não foi executada uma nova avaliação pycocotools contra as anotações COCO originais nesta revisão.

Fontes: `data/yolo11n-jetson24-20261116/{README.md,results.json,summary.csv,accuracy.csv}`, `raw/run-default-dvfs/`, `src/bench.py` e `src/analyze.py`. Conferência reproduzível: `tmp/jetson-review/auditoria_brutos.py` e `auditoria_brutos.txt`.

## Resultados que merecem destaque

Tabela principal sugerida: **TensorRT, lote 1, modo de potência 25 W, DVFS padrão, uma rodada por configuração**. A potência na tabela é medida em VDD_IN; 25 W é a configuração do modo, não a potência observada.

| Precisão TensorRT | Vazão e2e (inf/s) | Latência média (ms) | p99 (ms) | Potência (W) | Energia total (mJ/inf) | Energia incremental (mJ/inf) | mAP50–95 COCO (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 36,56 | 27,35 | 28,44 | 7,706 | 210,78 | 80,51 | 41,192 |
| FP16 | 44,87 | 22,28 | 23,30 | 6,814 | 151,86 | 45,72 | 41,238 |
| INT8 com fallback FP16 | 50,32 | 19,87 | 20,77 | 6,381 | 126,81 | 32,17 | 35,931 |

O mAP é medido nas 500 imagens COCO; desempenho e energia são medidos nos quadros do carrinho. A tabela relaciona propriedades da mesma configuração obtidas em cargas distintas, sem sugerir que todas as métricas vieram da mesma sequência.

Comparações descritivas verificadas:

- **FP16 frente a FP32:** vazão 1,227×; latência média 18,5% menor; energia total 28,0% menor; energia incremental 43,2% menor. A diferença de mAP é +0,046 ponto percentual; não deve ser interpretada como melhora comprovada de qualidade.
- **INT8 frente a FP16:** vazão 1,121×; latência 10,8% menor; energia total 16,5% menor; energia incremental 29,6% menor. Perda de 5,307 pontos percentuais de mAP50–95 (cerca de 12,9% relativa).
- **Escopo da medição:** FP16 tem 4,50 ms em `compute`, 5,73 ms em `infer` e 22,28 ms em `e2e`. São ensaios separados, não parcelas aditivas. A etapa de inferência dentro de e2e leva 14,95 ms, diferente dos 5,73 ms do modo infer contínuo. O DVFS e a alternância CPU/GPU são relevantes; não atribuir toda a diferença e2e−infer ao pré/pós-processamento.
- **Potência isolada não basta:** ORT CPU FP32 consome 8,07 W no ensaio e2e, mas entrega só 7,29 inf/s e gasta 1.107,39 mJ/inf. Comparar apenas watts esconderia a diferença no trabalho concluído.
- **FP16 não é um ganho universal:** no TorchScript, a latência e2e observada cresce de 27,13 para 28,75 ms, embora a energia total caia de 250,80 para 221,17 mJ/inf. A conclusão deve ser sobre configuração/runtime, não apenas extensão de arquivo ou tipo numérico.

O F1 de 0,9985 em FP16 e 0,6715 em INT8 nos quadros do carrinho é **concordância com ORT CPU FP32**, não F1 contra anotação humana. O algoritmo guloso em `src/analyze.py` não procura o segundo candidato livre quando o melhor já foi associado; pode subcontar correspondências. Essa métrica pode ficar no apêndice, explicitamente como concordância e com a limitação do pareamento. O mAP COCO é a evidência principal de degradação de qualidade nesta rodada.

Há uma rodada principal de 30 s após 5 s de aquecimento por configuração/modo, sem replicações independentes suficientes para estimar incerteza entre rodadas. Os percentis descrevem chamadas dentro da rodada. As notas em `results.json` relatam variação e2e em teste preliminar sob DVFS; diferenças pequenas entre runtimes devem ser tratadas como observações desta rodada. Não transformar o limiar informal de aproximadamente 20% dessas notas em teste estatístico.

## Como organizar o texto e as figuras

1. **Materiais e métodos:** acrescentar uma seção curta de caracterização da Jetson: equipamento/software, modelo COCO e hashes registrados, lote e forma da entrada, nove configurações, definição dos três escopos, aquecimento/duração, leitura INA3221 a 50 Hz, janelas ociosas, fórmulas de energia e protocolo de avaliação/calibração. Registrar que os serviços de monitoramento permaneceram ativos e que o serviço de produção foi parado antes da etapa de desempenho/energia. Evitar prometer múltiplos modos de potência se só há 25 W medido.
2. **Resultados, após treinamento:** substituir o trecho provisório da seção `sec_res_jetson` por “Desempenho e consumo do computador de bordo”. Abrir com a distinção entre modelo COCO de bancada e modelo MBG. Apresentar a tabela TensorRT, o compromisso qualidade/energia e a comparação dos escopos. Mostrar as nove configurações em uma figura de visão geral ou apêndice.
3. **Ligação com a missão:** acrescentar uma subseção “Implicações para a demanda de processamento”, com as razões condicionais abaixo. Reservar os resultados definitivos de viabilidade, consumo da estratégia e recall de focos para o pipeline MBG completo.
4. **Conclusões:** resumir o que foi medido e o que permanece condicionado. O arquivo de conclusão está vazio e sua inclusão está comentada no documento principal; não convém habilitá-lo antes de redigi-lo.

Figuras candidatas geradas em `tmp/jetson-review/`, com PNG para inspeção e PDF vetorial para uso no LaTeX:

- `plots_qualidade_energia`: escolha principal, pois torna visível o ganho energético e a perda de mAP do INT8. Não usar só um ranking de FPS.
- `plots_escopos_trt`: comparação das latências `compute`, `infer` e `e2e`, com barras agrupadas. Não empilhar esses modos; não são componentes de uma mesma execução.
- `plots_latencia_energia`: visão das nove configurações, energia total versus latência e2e. Boa para o apêndice ou como figura secundária; a escala logarítmica deve ficar explícita na legenda.

Evitar quatro gráficos de barras quase equivalentes para FPS, latência, energia e FPS/W. FPS/W é o inverso da energia total por inferência, após conversão de unidades, e acrescenta pouca informação ao gráfico de energia. Preferir uma tabela completa no apêndice e duas figuras com perguntas distintas.

## Ligação quantitativa com os resultados geométricos existentes

O capítulo já define energia adicional na Equação `eq_pc`. A identificação correta, neste ensaio, é:

\[
E_{\mathrm{total}}=\bar P/\mathrm{FPS},\qquad
E_{\mathrm{inf}}\approx E_{\mathrm{incremental}}=(\bar P-P_{c0})/\mathrm{FPS}.
\]

Usar energia total em `P_c = P_c0 + r_exec E_inf` contaria o consumo ocioso duas vezes. Mesmo usando a incremental, extrapolar de carga contínua para baixa cadência exige verificar a relação entre potência e taxa; DVFS e custos da estratégia podem alterar a energia por inferência.

Como exercício de dimensionamento, usar `rho = r_solicitado / FPS_bancada`. **Não chamar essa razão de utilização EDF, utilização de GPU ou garantia de prazo.** A Equação `eq_utilizacao` usa tempos de execução e hipóteses de escalonamento diferentes. O próprio texto, em `USPSC-Cap2-Desenvolvimento.tex:513`, exige medição do processamento completo, filas e descarte.

| Demanda geométrica | Recortes/s | rho usando TRT FP16 e2e como aproximação |
|---|---:|---:|
| Memória adiada, B–M | 5,40–8,86; mediana 6,74 | 0,120–0,197; mediana 0,150 |
| Memória adiada, A | 9,17–10,71 | 0,204–0,239 |
| Amostragem de 1 quadro/s, B–M nominal | 24 | 0,535 |
| Memória imediata, B–M, mediana | 159,93 | 3,564 |
| Todos os recortes de todos os quadros, B–M, valor aproximado do texto | 719 | 16,023 |

Esses números sugerem que a seleção adiada reduz a demanda a uma ordem de grandeza compatível com a capacidade observada em bancada. Não comprovam que a implementação completa cumpra os prazos. A chegada de recortes em rajadas e a decisão de analisar na última oportunidade exigem simulação/medição temporal: cobertura geométrica com execução instantânea não garante cobertura após filas e atrasos.

Somente sob a hipótese adicional de energia incremental constante, a demanda mediana de 6,74 recortes/s resultaria em `4,76252 + 6,74312 × 0,04572335 = 5,071 W`; a demanda de 24 resultaria em 5,860 W. A diferença de 0,789 W é ilustrativa, não uma economia medida na missão. A redução de aproximadamente 107× no número de passadas já relatada no capítulo **não equivale** a 107× de economia da placa ou da aeronave. Para demandas acima da capacidade, não calcular potência substituindo indiscriminadamente a taxa solicitada: o modelo pede a taxa efetivamente concluída.

Reprodução: `tmp/jetson-review/cenarios_missao.py`, `cenarios_calculos.json` e `cenarios_missao_condicionais.csv`. As linhas inviáveis por capacidade média aproximada ficam sem projeção de potência.

## Ajustes editoriais e próximos ensaios

No texto atual:

- `USPSC-Cap4-Resultados.tex:7` e `:140`: atualizar o futuro “reunirá” e o aviso “em elaboração” somente ao incorporar a seção; trocar “modos de potência avaliados” por “configurações avaliadas no modo de 25 W”.
- `USPSC-Cap4-Resultados.tex:133`: substituir “três causas identificáveis” por “diferenças de protocolo que podem contribuir”. Não houve ablação para identificar a contribuição causal de tamanho do modelo, número de voos e treinamento multiclasse. “Supera” o Faster R-CNN deve ser qualificado como valor numericamente maior, sob avaliações não equivalentes. A ressalva já presente ao fim do parágrafo é correta, mas deve orientar também as frases anteriores.
- Manter separados mAP COCO, AP/F1 por classe MBG, concordância entre implementações e recall de focos únicos. Nenhuma dessas métricas substitui automaticamente a outra.

Para fechar a evidência do TCC, priorizar:

1. Exportar os mesmos pesos MBG escolhidos na validação para FP32/FP16; avaliar a preservação de AP e recall por classe, especialmente objetos pequenos. Se incluir INT8, calibrar com recortes exclusivamente do treinamento, representativos do domínio, e escolher configuração na validação. Preservar o teste final até fixar o protocolo.
2. Medir o pipeline completo sobre as mesmas sequências de validação e os mesmos recortes: decodificação, seleção, rede, pós-processamento e associação. Registrar taxa solicitada/concluída, latência, fila, descarte e detecção de focos. Incluir os recortes vazios na reprodução da missão.
3. Repetir cada configuração em blocos independentes com ordem alternada ou aleatória. Usar replicações como unidade para a incerteza entre rodadas; as milhares de chamadas dentro de 30 s não são milhares de experimentos independentes.
4. Medir potência em cadências representativas da memória e da amostragem, além da saturação, para verificar o ajuste `P_c(r)`. Quantificar o ocioso do próprio pipeline com modelo carregado. Se a comparação entre modos de potência permanecer no escopo, repetir o protocolo nos modos escolhidos.
5. Só então integrar consumo por área e detecção à análise de H1–H3. O ensaio atual não mede efeito de massa na propulsão, economia global por alteração de altitude/velocidade nem a velocidade ótima condicionada à detecção.

## Parágrafo sugerido para abrir a seção

> A caracterização inicial do computador de bordo comparou nove configurações de execução do YOLO11n pré-treinado no COCO, com lote unitário e entrada de 640 × 640 pixels, no modo de potência de 25 W e com controle dinâmico de frequência. Esse ensaio identifica configurações candidatas para a implementação embarcada; ele não utiliza os pesos ajustados à MBG-V2 apresentados na seção anterior. No processamento desde a imagem já decodificada até as detecções, TensorRT FP16 apresentou latência média de 22,28 ms e energia total de 151,86 mJ por inferência, reduções de 18,5% e 28,0% em relação a TensorRT FP32 nesta rodada. O mAP50–95 no subconjunto COCO permaneceu próximo de 0,412. A configuração INT8 com fallback FP16 reduziu a energia total para 126,81 mJ, mas o mAP caiu para 0,359, evidenciando a necessidade de avaliar conjuntamente custo computacional e qualidade da detecção.
