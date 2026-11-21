# Revisão independente dos dados — 15/11/2026

## Resultados reproduzidos

- `USPSC-Cap3-MateriaisMetodos.tex:197,188,249`: leitura independente dos 26 XML, com `stop_frame+1`, caixas `outside != 1`, incluindo oclusas: 193.901 quadros, 1.649.228 caixas e 8.441 trilhas; mediana de 210 caixas/trilha. Presença por voo: mínimo 66,4418%, mediana 98,5511%; no conjunto, 90,5204% dos quadros têm pelo menos dois objetos e 36,6935% têm 6–10. Os arredondamentos do texto conferem. A mediana por voo e o agregado usam denominadores distintos, conforme corretamente explicado na figura.
- `:206-219`: contagens por classe, número de voos e regiões e medianas de lado em pixels conferem independentemente. Percentual menor que 32 pixels total é 10,6622%; garrafa 92,75%, balde 66,59% e vaso 31,75%. As demais porcentagens são compatíveis com arredondamento; `small_trash_bin` é exatamente 5/40 = 12,5%, exibido como 12%, e convém explicitar a regra de arredondamento para empates.
- `:17`: a soma dos campos `area` do GeoJSON é 1.373.416,083 m² = 1,373416 km², arredondando para 1,37 km². Isto verifica a soma dos atributos fornecidos, não valida independentemente a geometria nem o método de cálculo das áreas.
- `:267`: a conversão J/m² para Wh/km² por 10⁶/3600 está dimensionalmente correta. Em `Cap1:Justificativa`, 7/180 e 25/180 dão 3,89% e 13,89%; 3840×0,014 = 53,76 m, 2160×0,014 = 30,24 m e 4/30,24 = 13,23%, coerentes com os valores ilustrativos.

Reprodução das contagens: `python tmp/revisao/terra-dados.py` (Python padrão; executado na raiz). O programa percorre os XML originais, não usa os CSV de resultados. A comparação está limitada ao conteúdo dos XML e não prova que objetos únicos foram corretamente anotados.

## Pontos a esclarecer e limites

- `:289-294`: a cadeia independente de bateria e seus resultados são detalhados abaixo. O processamento original completo não foi localizado; recomenda-se preservá-lo com os filtros e extremos explícitos.
- `:108-146`: GSD do voo 03 e trilha 96 foram recalculados diretamente, conforme detalhado abaixo; demais 25 ajustes não foram recalculados. O resultado não valida a sincronização física ou escala absoluta.
- `:245-247`: não foi reproduzida a união das faixas, as sobreposições ou densidades; 1,42 km² de áreas imageadas não deve ser confundido com os 1,37 km² dos polígonos. As estimativas dependem de GSD, yaw e sincronização.
- `:280-281`: igualdade dos cabeçalhos, decodificação do último quadro e igualdade pixel a pixel dos PNGs não foram testadas nesta revisão.
- `Cap1`: estatísticas de dengue, LIRAa e estudos externos não foram verificadas nas fontes primárias nesta revisão. As fórmulas relevantes do Cap2 foram consultadas quanto a unidades; não foi feita auditoria completa da bibliografia ou física.
- Dimensões em centímetros, GSD dos demais voos, durações das trilhas, velocidades, percentis de altitude, distâncias percorridas e consumo em pontos percentuais não foram reproduzidos independentemente aqui.

## Reprodução complementar: GSD voo 03

Script independente `tmp/revisao/terra-dados-gsd.py`: lê XML e CSV, calcula coordenadas locais WGS84, interpola posição e guinada nos quadros a 29,97 fps, projeta centros das caixas e calcula as medianas. Há 292 trilhas e 13.228 amostras; a busca inicial 0,80–2,20 em passos de 0,05 e o refinamento 1,30–1,40 em passos de 0,01 têm mínimo em 1,35 cm/pixel, dispersão 0,547529526 m. A diferença máxima contra todos os pontos do CSV da curva é 4,95×10⁻⁹ m, consistente com arredondamento. Para GSD zero, 7,466615831 m.

Trilha 96: 45 amostras, quadros 3735/3845/3955, duração 7,340674 s, deslocamento líquido da aeronave 30,850778 m, do centro na imagem 2.112,1847 pixels, e das projeções extremas 2,790107 m; dispersão 0,547680 m. Confere com texto e metadados. A reprodução assume a sincronização do texto, yaw em décimos de grau e solo plano; não mede acurácia absoluta nem comprova a ausência de defasagem. A escolha da trilha foi inspecionada no script original; o script independente reproduz os valores da trilha escolhida, não a seleção entre candidatas.

## Reprodução complementar: bateria e energia

`tmp/revisao/terra-dados-energia-99.py` lê os 26 CSV, exclui linhas com `currentVoltage`, `currentCurrent` ou `remainPowerPercent` zerados, conserva primeiro tempo duplicado, ordena, usa tempo relativo ao primeiro registro válido e recorta ao intervalo comum com vídeo, cujo fim é `nb_frames/fps`; integra potência `-currentVoltage*currentCurrent/10⁶` por trapézios. Interpola o extremo quando há amostras que o envolvem; não extrapola a última amostra. É uma reconstrução independente plausível, não identificação do processamento original. O fator 10⁶ decorre da interpretação dos campos como mV e mA; os valores de tensão coincidem numericamente com `voltage(v)` após divisão por 1.000.

Resultados: 97.280 linhas brutas, 99 removidas e 713 tempos duplicados após remoção, total 812/97.280 = 0,835%; as 99 ocorrem nos cinco voos citados. A potência mediana nos voos B–M varia de 172,77 a 190,14 W, mediana 181,75 W; voo A 169,69 e 172,41 W. Energias B–M 25,566 a 68,020 kJ, soma de todos 330,205 Wh. Os números reproduzem os arredondamentos do texto. Nos voos 14 e 22 os últimos registros válidos terminam 0,81 e 9,82 s antes do fim do vídeo, respectivamente.

**Precisão necessária sobre filtro:** das 99 linhas excluídas, 57 têm corrente e tensão zeradas; as 42 restantes, todas no voo 06, têm `remainPowerPercent=0`, apesar de corrente e tensão não nulas. A frase “campos da bateria valem zero” é ambígua e deve discriminar o campo e a regra “qualquer um destes campos”. Mantendo essas 42 linhas e excluindo apenas tensão/corrente zero, a mesma integração resulta em 330,2037 Wh, contra 330,2050 Wh: diferença de 0,0013 Wh no total, sem alterar o arredondamento, mas com possível efeito em outros usos da telemetria. A regra do processamento original não pôde ser comprovada apenas pela coincidência da contagem.

O tratamento dos 713 tempos duplicados confere por contagem; a regra de manter a primeira linha é uma hipótese reproduzida, não comprovada a partir do resultado agregado. O fim `nb_frames/fps` é uma convenção; integração alternativa no tempo do último quadro pode diferir ligeiramente. A energia por área não foi validada, pois depende das áreas e GSD específicos dos 26 voos.

## Rastreabilidade das áreas

Busca em `code` e `doc/dados` encontrou rotina e metadados do GSD somente para voo 03 e não encontrou tabela dos 26 GSD, máscaras de união de áreas nem cadeia que gere os valores de área e densidade do texto. Não foi possível reproduzir os 1,42 km², 0,028–0,116 km², sobreposições ou 239 Wh/km² sem inventar parâmetros. Recomenda-se disponibilizar por voo GSD, telemetria tratada, instantes incluídos, geometria da grade, união e somas por região, com script e identificadores/hash de entrada. O denominador deve ser a união imageada, não soma de áreas dos quadros ou área de polígono; para a soma regional deve-se deduplicar a interseção dos dois voos.
