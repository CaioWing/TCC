# Revisão do TCC para envio à banca

**25 de novembro de 2026.** Cada uma das 19 seções numeradas dos capítulos 1 a 4 teve um revisor Astra responsável. A conclusão, os resumos em português e inglês, as listas, as referências e a legibilidade dos gráficos receberam revisões adicionais. A conferência usou as obras citadas, os arquivos da MBG-V2, as saídas do detector, o código de avaliação e os registros da bancada. Os voos, o treinamento e as medições na Jetson são os do estudo original.

## Conferência por seção

| Seção | Conferência realizada |
| --- | --- |
| 1.1 Problema e delimitação | Os totais de dengue correspondem aos casos **prováveis** informados pelo Ministério da Saúde. A introdução distingue a primeira detecção no vídeo do recebimento do alerta em solo. |
| 1.2 Objetivos | Os quatro objetivos específicos têm correspondência nos resultados. A comparação entre detecção e energia usa a faixa comum de 40 a 80 m. |
| 2.1 Potência em pairado | A velocidade na esteira, a tração e a potência ideal conferem com Glauert, pp. 202–203, e Leishman, pp. 62–63. A aproximação para a variação de massa mantém fixas a densidade e a área dos discos. |
| 2.2 Voo à frente | A relação implícita da velocidade induzida foi cotejada com Glauert, relatório 1111, §3, eq. 11, e com a convenção de sinais de Leishman, pp. 93–94. As parcelas de perfil e induzida são somadas pelos rotores. O arrasto da aeronave entra uma única vez. |
| 2.3 Autonomia e alcance | A condição de tangência minimiza potência por velocidade em voo uniforme sem vento. A conversão de J/m² para Wh/km² multiplica por 10⁶/3.600. |
| 2.4 Geometria | Escala, campo de visão e borrão foram cotejados com Paine e Kiser, p. 75, e O’Connor, Smith e James, eq. 6. A projeção local foi conferida com Kayton e Fried, p. 25, e com as transformações no código. O tempo de visibilidade pressupõe uma passagem completa. |
| 2.5 Detecção | O avaliador por quadro calcula a AP pela média da precisão em 101 níveis de recall. Já a tabela de validação usa a integração por trapézios da Ultralytics 8.4.183. O texto explicita as duas implementações. |
| 2.6 Computação embarcada | A condição de prazo por quadro usa o pior tempo de execução. A potência incremental resulta de multiplicar inferências/s por J/inferência. A vazão média de bancada, sozinha, não determina o atraso em fila. |
| 2.7 Co-projeto | A formulação identifica as restrições de detecção e de capacidade. A avaliação de recipientes da MBG-V2 foi feita na estação de trabalho. O efeito do hardware sobre esse recall ainda depende de ensaio. |
| 3.1 Base | A tese de Wesley L. Passos, §3.6, identifica o DJI Phantom 4 Pro. A recontagem dos 26 XML encontrou 1.657.466 registros de caixa, dos quais 8.238 têm o objeto marcado como fora da imagem. O capítulo informa o total efetivamente analisado. |
| 3.2 Divisão e treinamento | As regiões são disjuntas entre os conjuntos. O máximo da mAP₅₀–₉₅ no histórico está na época 17. O treino terminou na época 27. A identificação dos pesos exportados confere com o checkpoint selecionado. |
| 3.3 Seleção de recortes | O orçamento é igual por voo e conhecido retrospectivamente para o intervalo. A origem da malha e a interpolação da telemetria estão explícitas. Cada análise temporal identifica se inclui todas as trilhas ou apenas as confirmadas. |
| 3.4 Bancada | Os registros distinguem as 95 imagens do ensaio de tempo e energia, as 500 de avaliação COCO e as 512 de calibração INT8. A cronometragem a partir da imagem decodificada termina após a supressão das caixas sobrepostas. |
| 4.1 Validação do detector | A comparação com Souza usa o maior F1 publicado na Tabela A.7 para o grupo correspondente. Os dois estudos usam protocolos de treinamento e avaliação diferentes. |
| 4.2 Detecção nos seis voos | A memória e o intervalo localizaram 1.079 e 1.054 das mesmas 1.924 referências. O exame das duas vistas distingue 1.401 oportunidades de reconhecimento e 1.312 trilhas distintas. |
| 4.3 Potência em cruzeiro | O erro absoluto médio de 4,20 W usa dez voos reservados a 4,1 m/s, com o mesmo peso por voo. A inclinação extraída da telemetria permaneceu acima da previsão nas duas campanhas. |
| 4.4 Área e visibilidade | Revisitou-se o cálculo de área única e a definição da duração anotada. Um ponto dentro da faixa pode continuar no campo de visão mesmo se o recipiente deixar de ser reconhecível. |
| 4.5 Jetson | O ensaio FP16 registrou 44,87 recortes/s e 45,72 mJ incrementais por recorte. Aplicar 23,30 ms a cada recorte testa a fila com serviço fixo no P99 da bancada. O resultado não estima um percentil de espera em voo. |
| 4.6 Energia por área | Refeitas as contas de transporte, repouso e inferência. Cerca de 24,286 W adicionais por 1.573,24 s correspondem a 10,61 Wh. As medianas por percurso são 270,3 e 172,3 Wh/km² para o kit ativo a 40 e 80 m. |
| 5 Conclusão e resumos | Os resultados e denominadores concordam nos dois idiomas e na conclusão. A redação distingue medidas de bancada, resultados nos vídeos e projeções de energia. Os seis voos também orientaram a escolha da estratégia. |

**Fontes de verificação.** As páginas e equações consultadas nas obras locais e os nomes dos arquivos de dados estão nos registros técnicos. [Fundamentação](revisao-astra/fundamentacao.md) · [Métodos](revisao-astra/metodos.md) · [Resultados](revisao-astra/resultados.md) · [Relatório original de Glauert](https://reports.aerade.cranfield.ac.uk/handle/1826.2/1434) · [Avaliador oficial COCO](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)

## Figuras e interpretação

- **Figura 19.** O novo painel mostra em quantas oportunidades o recipiente foi reconhecido nas duas vistas, só em uma delas ou em nenhuma. O painel de sobreposição informa o denominador de cada coluna. Assim, o leitor consegue ver por que repetir uma vista pode ajudar mesmo quando grande parte do terreno é a mesma.
- **Figura 24.** As inclinações observadas e previstas agora podem ser comparadas diretamente em graus para as duas campanhas. Os números foram conferidos com os registros de voo.
- **Figura 32.** Os dois painéis reúnem a energia estimada por área e a proporção de recipientes localizados. Na projeção de energia, a área cresce com a altura. Na simulação de detecção, os 1.924 recipientes continuam os mesmos. Esses dados não permitem calcular quantos recipientes seriam encontrados no terreno adicional.
- **Figura 17.** Cada classe conserva sua própria escala de recall, identificada na legenda. Uma escala única de 0 a 100% tornaria quase indistinguíveis as curvas dos baldes, cuja detecção cai de menos de 6% a zero.

## Versão para a banca

A nova epígrafe de Alberto Santos-Dumont foi transcrita diretamente da p. 96 da edição de 1918, com a grafia original. [Fac-símile, p. 96](https://upload.wikimedia.org/wikipedia/commons/6/64/O_que_eu_vi%2C_o_que_n%C3%B3s_veremos_%281918%29.pdf#page=96)

A ficha reemitida pela EESC foi conferida e incorporada após a folha de rosto. O título termina em “por drone”, Glauco Augusto de Paula Caurin consta como único orientador e o código é C352s. O nome do autor, o curso, o ano e as cinco palavras-chave correspondem ao TCC. [Orientação da EESC para o TCC](https://eesc.usp.br/graduacao/tcc.php?id=18070)

A declaração de uso de inteligência artificial generativa foi incluída após os agradecimentos, com uma página em português e outra em inglês. O autor identificou o Codex (OpenAI) como a ferramenta que o auxiliou na revisão da literatura, na revisão gramatical, no tratamento dos dados e na elaboração dos gráficos. A ficha catalográfica e os dois resumos registram 89 páginas, em concordância com a última página numerada do PDF. [Biblioteca da EESC](https://eesc.usp.br/biblioteca/post.php?catid=normatizacao&guid=67)

O total de 1.649.228 caixas com o objeto presente nos XML difere das 1.702.737 caixas publicadas na tese de Passos. A documentação consultada não explica toda a diferença. O método informa ambas as contagens e identifica qual sustenta a análise.
