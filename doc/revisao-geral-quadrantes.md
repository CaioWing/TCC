# Parecer sobre os resultados e a conclusão do TCC

## O que os seis voos permitem concluir

A 40 m, com 8.796 recortes por método, os quadrantes localizaram 1.079 dos 1.924 recipientes e o intervalo pareado localizou 1.054. Os quadrantes também geraram menos falsos alarmes em cada voo. Entre as 1.035 trilhas confirmadas na imagem pelos dois métodos, a mediana do tempo até o primeiro alerta visual foi de 1,33 s nos quadrantes e 2,37 s no intervalo. As três medidas respondem a perguntas diferentes. A localização exige um acerto no mapa, os falsos alarmes contam focos sem correspondência e o tempo usa a primeira detecção correta na imagem.

O saldo de 25 localizações esconde diferenças importantes. Os dois métodos localizaram 909 recipientes em comum. Outros 170 foram localizados somente pelos quadrantes e 145 somente pelo intervalo. Dos 25 acertos de saldo, 22 vieram dos voos 16 e 23. As caixas d’água contribuíram com um saldo de 27. Nas demais classes, consideradas em conjunto, o intervalo localizou dois recipientes a mais. Nas 216 trilhas com maior variação do movimento, as confirmações em até cinco segundos favoreceram o intervalo por cinco trilhas. Esses dados delimitam a conclusão sobre mudanças de direção ou velocidade.

## O que mudaria na operação em voo

**Origem da malha.** Na reconstrução, a origem dos pontos espaçados de 8,64 m é a posição média dos registros válidos de GPS do voo inteiro. A posição da câmera em cada quadro vem da interpolação da telemetria. Para a operação a bordo, é preciso definir a origem antes dos envios e trabalhar com as poses disponíveis naquele instante. Em uma verificação geométrica, fixar a malha na primeira posição de cada voo alterou o total de 8.796 para 8.920 recortes. Coincidiram 160 pares de quadro e posição de recorte. Dois quadros próximos podem mostrar imagens semelhantes, por isso essa diferença geométrica, sozinha, não permite calcular a mudança de recall ou de tempo até o primeiro alerta.

**Escolha da estratégia.** As regiões de teste ficaram fora do treinamento do detector. Os resultados desses mesmos seis voos, porém, ajudaram a escolher qual regra de memória destacar. O intervalo de comparação também foi definido retrospectivamente, usando a cota final de cada voo. O teste de generalização exige fixar a regra e a origem da malha antes de aplicá-las a voos ou regiões ainda não usados nessa escolha.

**Tempo real e energia.** A vazão e a energia por recorte foram medidas em bancada. A fila usa os horários reais dos pedidos, e a energia por área combina telemetria dos voos sem o kit com o custo estimado da carga e da placa. Verificar o sistema completo exige cronometrar da captura 4K à emissão do alerta, com os pesos da MBG-V2 a bordo, e medir a bateria em percursos comparáveis com e sem o kit. A integração permanece relevante mesmo com a demanda média de 5,59 recortes/s. Os ensaios de precisão numérica mediram a qualidade no COCO. O recall dos voos foi medido em TensorRT FP16.

**Altura.** Os ensaios a 50, 60, 70 e 80 m equivalentes preservam as trilhas, os quadros e as posições dos recortes de 40 m. Permitem analisar a perda de detalhe e a alteração do contexto visual. Um voo real em outra altura muda também o terreno visível e os pedidos da memória.

## Interpretação do modelo físico

A razão entre energia medida e prevista e o fator de revisitas usam a mesma área reconstruída, e esse fator integra a própria razão. A correlação de 0,98 apresentada na seção 4.4.1 ajuda a descrever a decomposição. Para verificar a área por uma medida independente, seria necessário obter a área por outra fonte. Na seção 4.4.2, as trilhas também contribuem para ajustar o GSD usado na estimativa de visibilidade. A comparação com suas durações verifica a consistência interna. O erro da curva de potência nos voos sem carga também deve permanecer distinto da validação do custo adicional dos 174 g.

## Pendências para a versão de depósito

O Resumo e o Abstract ainda contêm os avisos provisórios do modelo. Podem ser redigidos a partir dos resultados consolidados. A ficha catalográfica e a folha de aprovação exibem as instruções do modelo e precisam dos documentos institucionais definitivos na etapa de depósito.
