# Quadrantes com origem na primeira posição de GPS

Nos seis voos completos a 40 m, o intervalo pareado localizou 13 recipientes a mais. Com a mesma cota em cada voo, os quadrantes geraram 432 falsos alarmes a menos e a mediana do tempo entre a primeira aparição e a detecção correta foi menor tanto no conjunto das trilhas quanto em cada voo isolado.

## Comparação com o ensaio anterior

| Origem usada para definir a cota | Método | Recortes | Localizados em 1.924 | Recall no mapa | Falsos alarmes no mapa | Alertas até 1 s | Mediana do primeiro alerta | P90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Média do voo | Quadrantes | 8.796 | 1.079 | 56,08% | 1.291 | 416 | 1,37 s | 4,31 s |
| Média do voo | Intervalo pareado | 8.796 | 1.054 | 54,78% | 1.726 | 228 | 2,40 s | 4,75 s |
| Primeiro GPS | Quadrantes | 8.925 | 1.060 | 55,09% | 1.301 | 446 | 1,33 s | 4,34 s |
| Primeiro GPS | Intervalo pareado | 8.925 | 1.073 | 55,77% | 1.733 | 242 | 2,44 s | 4,67 s |

Com a primeira posição, a memória enviou 129 recortes a mais que com a média do voo, elevando a demanda de 5,59 para 5,67 recortes/s. Localizou 19 recipientes a menos. Dos 1.924 anotados, 931 foram localizados com ambas as origens, 148 somente com a média e 129 somente com o primeiro GPS. Ao ajustar o intervalo à nova cota, o controle temporal localizou 19 recipientes a mais. O saldo entre as estratégias passou de 25 localizações a favor dos quadrantes para 13 a favor do intervalo.

## Resultados por voo com o primeiro GPS

| Voo | Recortes por método | Recipientes anotados | Localizados, Q / I | Recall, Q / I | Falsos alarmes, Q / I | Mediana do primeiro alerta, Q / I |
|---|---:|---:|---:|---:|---:|---:|
| 09 | 918 | 205 | 125 / 124 | 60,98% / 60,49% | 74 / 112 | 1,50 s / 2,27 s |
| 10 | 854 | 205 | 150 / 149 | 73,17% / 72,68% | 65 / 122 | 1,20 s / 2,20 s |
| 15 | 1.545 | 467 | 156 / 158 | 33,40% / 33,83% | 432 / 497 | 1,27 s / 2,37 s |
| 16 | 1.531 | 253 | 140 / 132 | 55,34% / 52,17% | 245 / 320 | 1,33 s / 2,54 s |
| 23 | 2.066 | 440 | 299 / 306 | 67,95% / 69,55% | 300 / 431 | 1,32 s / 2,44 s |
| 24 | 2.011 | 354 | 190 / 204 | 53,67% / 57,63% | 185 / 251 | 1,43 s / 2,60 s |

Q indica quadrantes e I, intervalo pareado. Entre os 1.051 recipientes que os dois métodos confirmaram na imagem com a nova cota, os quadrantes alertaram primeiro em 692, o intervalo em 321, e houve 38 empates. Nesse grupo comum, as medianas foram de 1,30 s e 2,37 s, respectivamente.

## Como interpretar o ensaio

O recall conta a fração dos recipientes anotados que receberam uma localização correta no mapa. Um falso alarme é um ponto final no mapa que ficou sem correspondência única com uma referência da mesma classe a até 1 m. O tempo é contado na imagem da primeira aparição anotada de cada recipiente até a primeira detecção correta. A mediana e o P90 usam as trilhas confirmadas por cada método. A coluna de alertas até 1 s conta recipientes dentre todos os 1.924.

A primeira posição válida e a sua latitude definem a origem e a escala local da conversão de GPS em metros para a seleção. O teste geométrico preliminar, que apenas transladava as posições já convertidas com a latitude média, dava 8.920 recortes. Usar o referencial do primeiro GPS em ambas as etapas produziu 8.925. Todas as métricas do mapa mantêm o referencial comum do ensaio anterior para que o pareamento entre detecções e anotações seja comparável. A malha de 8,64 m, as regras de primeira vista de cada quadrante e a orientação da câmera são as mesmas. O detector treinado na MBG-V2 foi executado em TensorRT FP16 nas imagens originais. Nos 2.689 recortes comuns aos dois ensaios, considerando a união das duas estratégias em cada ensaio, as saídas do detector foram idênticas.

O intervalo começa no quadro zero e distribui pela duração de cada voo o total efetivamente solicitado pelos quadrantes naquele voo. A comparação é retrospectiva e a posição de cada quadro ainda usa a interpolação da telemetria. Estes tempos medem o vídeo e excluem o processamento a bordo.

Para reproduzir a partir da raiz do projeto, execute `python code/avaliacao_mbg/quadrantes_primeiro_gps.py`. As tabelas preservam as duas origens e identificam o método em cada linha. [Resultado agregado](resumo.csv) · [Resultados por voo](por_voo.csv) · [Resultados por trilha](por_trilha.csv) · [Tempos nas trilhas comuns](tempos_pareados.csv) · [Protocolo](protocolo.json).
