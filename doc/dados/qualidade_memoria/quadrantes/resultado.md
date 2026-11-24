# Memória por passagem e quadrante nos seis voos completos

Nos vídeos originais a 40 m, a câmera enquadra pontos do terreno espaçados de 8,64 m. Cada ponto pede um recorte ao entrar na imagem e ao aparecer pela primeira vez em cada um dos outros quadrantes. Os quadrantes são as quatro regiões formadas pelos eixos horizontal e vertical que passam pelo centro do quadro. Quando o ponto sai da imagem, termina a passagem. Se voltar a aparecer, pode pedir uma nova vista.

A regra solicitou 8.796 recortes nos seis voos. Todos os pedidos puderam ser atendidos no quadro em que ocorreram. Para comparar as estratégias com esse custo, o controle temporal distribuiu o mesmo número de recortes em instantes regulares de cada voo, com fase inicial zero e um quadro parcial quando a cota não era múltipla de 24.

| Método | Recortes | Localizados entre 1.924 | Recall no mapa | Falsos alarmes no mapa | Alerta visual em até 1 s | Mediana do primeiro alerta | P90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Quadrantes | 8.796 | 1.079 | 56,08% | 1.291 | 21,6% | 1,37 s | 4,31 s |
| Intervalo com a cota dos quadrantes | 8.796 | 1.054 | 54,78% | 1.726 | 11,9% | 2,40 s | 4,75 s |
| Memória por cobertura a cada 20 quadros | 37.776 | 1.220 | 63,41% | 2.903 | 51,4% | 0,53 s | 1,97 s |
| Intervalo de 30 quadros | 37.776 | 1.263 | 65,64% | 3.118 | 55,8% | 0,63 s | 2,00 s |

O tempo de alerta é medido no relógio do vídeo, desde a primeira aparição anotada até a confirmação visual. A fração em até um segundo considera todas as 1.924 trilhas. A mediana e o percentil 90 consideram as trilhas confirmadas por cada método. Restringindo a comparação às 1.035 trilhas confirmadas pelas duas estratégias de menor custo, as medianas foram de 1,33 s para os quadrantes e 2,37 s para o intervalo pareado.

| Voo | Recortes por método na comparação pareada | Localizados pelos quadrantes | Localizados pelo intervalo pareado | Falsos dos quadrantes | Falsos do intervalo pareado |
|---|---:|---:|---:|---:|---:|
| 09 | 887 | 128 | 126 | 69 | 122 |
| 10 | 843 | 150 | 149 | 78 | 120 |
| 15 | 1.551 | 158 | 158 | 448 | 499 |
| 16 | 1.508 | 140 | 129 | 239 | 292 |
| 23 | 2.015 | 305 | 294 | 281 | 438 |
| 24 | 1.992 | 198 | 198 | 176 | 255 |

Com a mesma quantidade de recortes, os quadrantes localizaram 25 recipientes a mais e geraram 435 falsos alarmes a menos. O ganho de localização concentrou-se nos voos 16 e 23, com 11 recipientes a mais em cada um. Frente ao intervalo de 30 quadros, a regra reduziu a demanda média de 24,01 para 5,59 recortes/s, com queda de 9,56 pontos percentuais de recall.

O ensaio pode ser reproduzido na raiz do projeto com `python code/avaliacao_mbg/quadrantes_voos.py`. Os limiares, o teto de recortes e a regra de desempate constam em [protocolo.json](protocolo.json). Os valores antes do arredondamento estão em [resumo.csv](resumo.csv), [por_voo.csv](por_voo.csv) e [por_trilha.csv](por_trilha.csv).
