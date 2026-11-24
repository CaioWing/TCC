# Escala dos recipientes e inferência integral

Resultados dos seis voos completos, com 1.924 referências das seis classes. Cobertura e intervalo usaram 37.776 recortes por método e condição. Os 40 m correspondem às imagens originais. As alturas maiores representam uma simulação do tamanho aparente nas mesmas regiões e nos mesmos instantes.

| Método | Altura (m) | Localizados | Recall (%) | Δ recall vs. 40 m (p.p.) | Falsos alarmes | Alertas até 1 s | Mediana do atraso (s) | P90 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Memória por cobertura | 40 | 1.220 | 63,41 | 0,00 | 2.903 | 989 | 0,53 | 1,97 |
| Memória por cobertura | 50 | 1.149 | 59,72 | -3,69 | 2.336 | 944 | 0,55 | 1,76 |
| Memória por cobertura | 60 | 1.127 | 58,58 | -4,83 | 2.132 | 899 | 0,57 | 1,97 |
| Memória por cobertura | 70 | 1.069 | 55,56 | -7,85 | 1.915 | 858 | 0,57 | 2,20 |
| Memória por cobertura | 80 | 1.020 | 53,01 | -10,40 | 1.840 | 813 | 0,57 | 2,30 |
| Integral | 40 | 1.400 | 72,77 | — | 5.892 | 1.212 | 0,07 | 1,31 |
| Um quadro a cada 30 | 40 | 1.263 | 65,64 | 0,00 | 3.118 | 1.073 | 0,63 | 2,00 |
| Um quadro a cada 30 | 50 | 1.175 | 61,07 | -4,57 | 2.455 | 1.012 | 0,67 | 1,97 |
| Um quadro a cada 30 | 60 | 1.151 | 59,82 | -5,82 | 2.247 | 968 | 0,67 | 2,00 |
| Um quadro a cada 30 | 70 | 1.094 | 56,86 | -8,78 | 2.017 | 901 | 0,70 | 2,13 |
| Um quadro a cada 30 | 80 | 1.052 | 54,68 | -10,97 | 1.924 | 851 | 0,70 | 2,29 |

A referência integral usa 1.131.600 recortes nos 47.150 quadros originais. O atraso é contado desde a primeira aparição anotada até a primeira confirmação visual. A mediana e o P90 usam as trilhas confirmadas por cada condição. A contagem até um segundo inclui todas as trilhas no denominador. Os falsos alarmes e o recall são avaliados no mapa ao fim do voo.

A simulação conserva a área pontuada e a seleção. Ela mede o efeito de escala e do contexto na imagem. O desempenho em um voo real a outra altura também depende da área nova que a câmera passa a ver, da trajetória e das condições de aquisição.
