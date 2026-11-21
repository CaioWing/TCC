# Revisão consolidada do TCC — 15/11/2026

## Escopo e resultado

Cinco subagentes GPT-6.1 Terra participaram: três revisaram a redação dos capítulos existentes (introdução, fundamentação e materiais e métodos), um revisou equações e um auditou os dados. A coordenação confrontou as sugestões com o texto, aplicou alterações selecionadas e solicitou conferências adicionais das equações e do procedimento de GSD após as edições.

O texto foi revisado quanto a naturalidade, coesão, redundâncias e precisão. Não se aplicaram indiscriminadamente todas as sugestões: foram preservadas as explicações solicitadas anteriormente, especialmente a ligação entre o modelo geométrico e o ajuste de GSD, a interpretação da dispersão, a referência pela mediana e o caso GSD = 0.

Este relatório registra o estado final. Os relatórios individuais são registros de análise e propostas, com referências às linhas da versão lida; suas sugestões não equivalem a alterações aplicadas. O relatório de equações contém uma retratação e alternativas posteriormente substituídas: prevalece a solução consolidada abaixo.

## Alterações aplicadas

- **Introdução:** ligação mais direta entre problema, hipóteses e objetivos; descrição da memória geográfica compatível com reobservações planejadas; distinção entre métricas de detecção e avaliação da missão; atualização da referência preliminar de cinco voos para os 24 voos das regiões B–M, com mediana das potências medianas próxima de 182 W.
- **Fundamentação:** explicação de projeção central antes da derivação do GSD; sequência das rotações antes da matriz; remoção de repetições sobre sinais dos deslocamentos e limitações da arfagem; distinção mais clara entre região imageada e inspeção com qualidade, entre formação de focos e avaliação, e entre demanda média de processamento e cumprimento de prazos.
- **Materiais e métodos:** abertura mais direta; conexão entre as equações das Seções 3.1.2.2 e 3.1.2.3; redução de repetições na explicação e na legenda da curva de GSD; transições entre densidade, sobreposição e quantidade de objetos por quadro; intervalos numéricos em prosa mais naturais; menos repetição das ressalvas sobre a campanha A.
- **Precisão dos dados:** percentual de lixeiras pequenas abaixo de 32 pixels explicitado como 12,5%; discriminação dos campos zerados nas 99 linhas de bateria, mantendo a ressalva de rastreabilidade do processamento original.

### Correção na formulação do coprojeto

A taxa de passadas solicitadas e a taxa efetivamente concluída não são necessariamente iguais. O texto agora distingue:

- `r(π,h,V)`: demanda da estratégia, usada na restrição de capacidade;
- `r_exec(π,h,V,H)`: passadas concluídas, usada no modelo de consumo;
- `r_max(H,π)`: capacidade medida para o processamento completo da estratégia;
- `R_f(h,V,π,H)`: taxa de detecção também dependente do hardware;
- `E_inf(H,π)`: energia incremental ajustada para hardware e estratégia.

A condição de admissibilidade continua comparando demanda e capacidade. Ela não garante, sozinha, o cumprimento dos prazos de cada quadro; latência, filas e descartes continuam necessários na avaliação. As simplificações energéticas que igualam demanda e execução foram condicionadas à conclusão de todas as passadas solicitadas. A Figura 8 continua representando demanda ideal, sem demonstrar capacidade do hardware.

## Resultados reproduzidos independentemente

| Verificação | Resultado | Limite da verificação |
| --- | --- | --- |
| Anotações dos 26 XML | 193.901 quadros, 1.649.228 caixas, 8.441 trilhas; mediana de 210 caixas por trilha | Confere o conteúdo dos arquivos, não a correção semântica das anotações |
| Objetos simultâneos | 90,5204% dos quadros com pelo menos dois objetos; 36,6935% com 6–10 | Agregado dos quadros, distinto da mediana das proporções por voo |
| Objetos pequenos | 10,6622% das trilhas abaixo de 32 pixels pelo critério adotado | Estatística em pixels; dimensões físicas dependem do GSD |
| Áreas fornecidas no GeoJSON | Soma de 1.373.416,083 m², equivalente a aproximadamente 1,37 km² | Soma dos atributos; geometria não recalculada |
| Curva de GSD do voo 03 | 292 trilhas, 13.228 amostras; mínimo em 1,35 cm/pixel e dispersão de 0,547529526 m | Resultado condicionado à sincronização e ao modelo de projeção adotados; não mede acurácia absoluta |
| Trilha 96 do voo 03 | 45 amostras; deslocamento da aeronave de 30,850778 m; deslocamento na imagem de 2.112,1847 pixels; separação das projeções extremas de 2,790107 m | Valores da trilha reproduzidos; seleção entre candidatas não reexecutada independentemente |
| Bateria dos 26 voos | Potências medianas B–M entre 172,77 e 190,14 W, mediana de 181,75 W; energias B–M entre 25,566 e 68,020 kJ; total de 330,205 Wh | Reconstrução plausível que reproduz os arredondamentos, sem comprovar a cadeia original |

As equações foram conferidas quanto a álgebra, sinais, unidades e coerência com as hipóteses declaradas. Não foram identificadas outras contradições materiais na formulação após os ajustes. Isso não substitui validação experimental dos modelos nem uma auditoria integral das fontes bibliográficas.

## Pendências prioritárias

1. **Rastreabilidade das áreas e dos GSD por voo.** A busca em `code` e `doc/dados` localizou o ajuste reproduzível do voo 03, mas não a tabela dos 26 GSD nem a cadeia completa de união das regiões imageadas. Permanecem sem reprodução independente os 1,42 km², os intervalos de área por voo, as sobreposições, as densidades por km² e os 239 Wh/km². Isso é uma lacuna de rastreabilidade, não demonstração de erro numérico. O próximo passo é recuperar ou reconstruir essa cadeia com os parâmetros efetivamente usados.
2. **Protocolo original da bateria.** A reconstrução reproduz os resultados arredondados, mas não identifica inequivocamente a regra original de filtragem, duplicatas e extremos. Das 99 linhas, 57 têm tensão e corrente zeradas; outras 42 têm apenas percentual de carga zero. Manter estas últimas muda o total reconstruído em cerca de 0,0013 Wh, sem alterar o arredondamento. Preservar o script original e seus parâmetros eliminaria essa ambiguidade.
3. **Elementos ainda incompletos do documento.** Resumo e abstract contêm conteúdo de modelo e precisam ser redigidos com o trabalho consolidado. Ficha catalográfica e folha de aprovação também precisam de conferência final. Dados pessoais e agradecimentos não foram reescritos.
4. **Verificações fora desta rodada.** Não foram refeitas a conferência integral das referências externas, as comparações pixel a pixel entre PNG e vídeo, a decodificação do último quadro, nem todos os cálculos de altitude, velocidade, distância e dimensões físicas. As conclusões desta revisão não devem ser estendidas a esses itens.

## Validação do documento

O PDF final foi compilado com sucesso, com 75 páginas físicas e sem referências indefinidas. A disposição dos capítulos revisados foi conferida visualmente, incluindo as figuras e a formulação do coprojeto. Permanecem avisos de composição já existentes, incluindo uma linha da bibliografia que excede a largura prevista; não foram tratados como falhas das alterações editoriais.

## Registros e scripts

- [Revisão da introdução](revisao-terra-introducao-2026-11-15.md)
- [Revisão da fundamentação](revisao-terra-fundamentacao-2026-11-15.md)
- [Revisão de materiais e métodos](revisao-terra-metodos-2026-11-15.md)
- [Auditoria de equações e discussão das correções](revisao-terra-equacoes-2026-11-15.md)
- [Auditoria de dados e limites de reprodução](revisao-terra-dados-2026-11-15.md)
- [Script independente das anotações](../tmp/revisao/terra-dados.py)
- [Script independente do GSD e da trilha](../tmp/revisao/terra-dados-gsd.py)
- [Script independente da energia](../tmp/revisao/terra-dados-energia-99.py)

A versão anterior à revisão foi preservada em `tmp/revisao/terra-baseline-2026-11-15`. Os scripts de auditoria são registros desta revisão e não substituem a cadeia de processamento original do trabalho.
