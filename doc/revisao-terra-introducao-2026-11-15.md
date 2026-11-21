# Revisão editorial — introdução e pré-textuais

Revisão de 15/11/2026. Escopo: leitura integral da introdução e dos elementos pré-textuais incluídos no arquivo principal; consulta às aberturas e aos encerramentos dos capítulos 2 e 3. As sugestões abaixo não alteram o TCC e não substituem a revisão de dados e equações.

## Achados prioritários

1. **Média — contabilidade da energia.** `doc/USPSC-TA-Textual/USPSC-Cap1-Introducao.tex:18`: “somando os consumos de energia da propulsão do multirrotor e do computador de bordo”. A frase omite as demais cargas, embora a linha 35 e o modelo integrado do capítulo 2 as incluam. Sugestão: “A métrica central é a energia gasta por quilômetro quadrado inspecionado, considerando a propulsão, os demais sistemas da aeronave e o computador de bordo.”

2. **Média — nível de comparação na justificativa.** `.../USPSC-Cap1-Introducao.tex:43`: “Esses dois aspectos não são descritos pelas métricas usuais de visão computacional.” A taxa de detecção é também uma métrica de visão computacional; a diferença relevante é sua apuração por objeto único e no contexto da missão. Sugestão: “As métricas usuais de visão computacional, calculadas sobre imagens, não descrevem por si só esses dois aspectos no nível da missão.” Preservar a explicação que segue sobre o acoplamento.

3. **Média — objeto potencial versus foco confirmado.** `.../USPSC-Cap1-Introducao.tex:14`: “quantidade de criadouros efetivamente encontrados na inspeção aérea de focos”. Antes da definição na linha 27, a frase pode sugerir confirmação de criadouros ativos. Sugestão: “quantidade de potenciais criadouros detectados na inspeção aérea de áreas com risco de reprodução do \textit{Aedes aegypti}”. Na linha 41, trocar “recipientes visíveis” por “objetos anotados” também aproxima o denominador da definição operacional da linha 18, sem afirmar que todos os objetos visíveis são anotados.

4. **Média — regra da memória geográfica.** `.../USPSC-Cap1-Introducao.tex:23`: “processar apenas a região do solo ainda não observada”. A exclusão de áreas já observadas conflita com o número desejado de observações por ponto na fundamentação (capítulo 2, linhas 537–541). Sugestão: “selecionar recortes com base no histórico estimado de observações das regiões do solo, estratégia aqui chamada de \textit{memória geográfica}”. Na linha 47, substituir “áreas ainda não observadas” por “áreas com menos observações do que o desejado”. Isso admite reobservações e preserva as hipóteses geométricas.

5. **Média — limite inclusivo da meta.** `.../USPSC-Cap1-Introducao.tex:18,53`: “acima de uma meta definida”. O capítulo 2, linhas 577–585, usa condição maior ou igual. Sugestão nas duas ocorrências: “igual ou superior a uma meta definida”. Mantém a mesma regra matemática expressa verbalmente.

6. **Baixa — comparação energética.** `.../USPSC-Cap1-Introducao.tex:45`: “um computador operando entre 7 e 25 W representa de 4% a 14% do consumo da aeronave”. O trecho compara potências, não consumos de energia. Além disso, o capítulo 3 já caracteriza os 24 voos das regiões B a M, com mediana de 182 W (linha 269), tornando dispensável a referência anterior a cinco voos. Sugestão para as duas frases: “Nos 24 voos das regiões B a M, a mediana das potências medianas por voo foi de 182 W (Seção~\ref{sec_base_energia}). Uma potência de computador entre 7 e 25 W equivale a cerca de 4% a 14% dessa referência.” Manter a explicação separada do efeito da massa e seu caráter hipotético.

7. **Baixa — frase longa no problema de pesquisa.** `.../USPSC-Cap1-Introducao.tex:53`: “e quanto essa recomendação depende das hipóteses do modelo de voo?” O referente singular pode ser pouco claro após “combinações”. Sugestão, preservando a questão: “Quais combinações de altitude, velocidade, estratégia de inferência e configuração de computação apresentam menor energia por quilômetro quadrado inspecionado, mantendo a taxa de detecção de focos únicos igual ou superior à meta definida? Quanto a escolha dessas combinações depende das hipóteses do modelo de voo?”

8. **Baixa — uniformidade temporal na descrição de H1.** `.../USPSC-Cap1-Introducao.tex:66`: “todas as estratégias usam” e “são processados” aparecem junto de “será fixada antes dos experimentos”. Como o procedimento ainda será executado, pode-se usar “todas as estratégias usarão” e “serão processados”, preservando o presente nas hipóteses como enunciados a testar. A caracterização da base e a análise preliminar continuam adequadamente no passado/presente.

## Pendências pré-textuais

- `doc/USPSC-TA-PreTextual/USPSC-Resumo.tex:11` e `.../USPSC-Abstract.tex:11` contêm marcadores explícitos de texto ainda não escrito. Ambos são incluídos pelo principal. Elaborar os textos ao final, a partir de métodos e resultados efetivamente obtidos; as palavras-chave já estão preenchidas.
- `doc/USPSC-TCC-modelo-EESC.tex:266,300` inclui PDFs de ficha catalográfica e folha de aprovação. Confirmar se são as versões definitivas adequadas ao estágio da entrega; a folha assinada depende da aprovação. Não é possível concluir a partir dos comandos que seu conteúdo esteja definitivo.
- Dedicatória, agradecimentos e epígrafe são textos efetivos, não marcadores do modelo; não identifiquei corte editorial material que justifique mexer em mensagens pessoais. As listas de siglas e símbolos também estão preenchidas; sua conferência técnica integral fica fora desta revisão editorial.

## Coerência geral

A introdução delimita o caráter potencial dos focos, diferencia o cenário de aplicação da avaliação em bancada e preserva as limitações de extrapolação. A estrutura anunciada para os capítulos 2 e 3 corresponde ao conteúdo atual: fundamentação e caracterização da base, com procedimentos experimentais ainda por acrescentar. Objetivo, pergunta e hipóteses são compatíveis em escopo; não há motivo editorial para retirar as explicações sobre massa, processamento e limitações. A organização dos capítulos 4 e 5 está explicitamente no futuro, coerente com sua ausência no arquivo principal. Além dos pontos acima, não encontrei redundância material que justifique cortes na introdução.
