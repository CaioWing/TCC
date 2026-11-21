# Revisão editorial — fundamentação teórica

Revisão de `USPSC-TA-Textual/USPSC-Cap2-Desenvolvimento.tex`, com consulta à introdução e aos materiais e métodos. As linhas referem-se ao estado de 15/11/2026. São propostas, não alterações aplicadas ao capítulo. Preservam-se as derivações de projeção central, fração de área nova, sinais, senos e cossenos e rotação, que cumprem função didática. Não se pressupõe erro nas equações.

## Maior impacto: precisão do fio condutor

1. **Linhas 205–214 e 562 — área imageada e inspecionada.** A fórmula usa uma taxa geométrica de área imageada, mas é apresentada como energia por área inspecionada antes de explicitar a condição de qualidade. Na linha 207, usar “A energia por unidade de área imageada é então”; na 212, “Para interpretá-la como energia por área inspecionada, exige-se que o processamento satisfaça o critério de qualidade adotado.” Na 562, usar “a energia por área imageada, tomada como inspecionada quando satisfeita a restrição de qualidade, é escrita...”. Manter o denominador comum e o tratamento das regiões não processadas, coerentes com o Capítulo 3.

2. **Linhas 532–547 — cadência fixa e ajustada.** A referência temporal ajustada à velocidade antecede a expressão “estratégias de taxa fixa”, que pode ser lida como incluindo essa referência. Substituir a abertura da linha 545 por: “Quando a cadência de quadros processados é mantida fixa ao variar a velocidade, a taxa de inferências também é fixa; nesse caso, a energia de computação por área, $P_c/(Vw)$, diminui com a velocidade. A referência temporal ajustada à velocidade, descrita acima, não obedece necessariamente a esse comportamento.” Manter a explicação da memória e suas hipóteses.

3. **Linhas 502–513 e 577 — capacidade e tempo real.** A seção distingue taxa média de prazo, mas “Uma configuração só é viável” pode sugerir que a restrição de taxa prova viabilidade temporal. Substituir a abertura da linha 577 por: “No modelo de otimização, uma configuração é admissível se atingir a taxa de detecção exigida e não exceder a capacidade média medida do hardware. O cumprimento de prazos requer ainda a análise de latências e filas descrita na seção de computação embarcada.” Preservar os teoremas e suas hipóteses.

4. **Linha 452 — detectores de estágio único.** A afirmação de que dispensar propostas “é o que torna esses detectores adequados ao processamento em tempo real” soa como garantia geral. Substituir por: “Ao dispensar uma etapa separada de propostas de regiões, essa arquitetura favorece a execução rápida; a capacidade de cumprir prazos depende do modelo, da entrada e do hardware.” Manter a referência a YOLO e os compromissos entre modelos.

5. **Linhas 203, 218 e 545–547 — velocidade de alcance.** A equivalência da linha 218 pode ser confundida com a curva anterior, que exclui o computador. Substituir sua primeira frase por: “Para uma largura de faixa fixa, minimizar $e$ equivale a minimizar $P_{el,\mathrm{tot}}(V)/V$: é a velocidade de alcance máximo da curva elétrica total correspondente à estratégia considerada.” Assim, a carga computacional pode variar com $V$.

6. **Linhas 454–459 — finalidade dos recortes.** “Para não reduzir os objetos pequenos a poucos pixels” sugere que recortes evitam uma perda já presente na imagem original. Substituir por: “Para preservar mais dos pixels disponíveis para objetos pequenos na entrada do detector, a imagem é dividida em recortes de lado $s_t$ pixels, cada um processado por uma passada da rede.” Manter a conta e as bordas.

## Clareza e coesão

7. **Linhas 128–132 — ressalva repetida sobre arfagem.** Após a lista e a explicação do vento, o último parágrafo repete forças adicionais e vento. Cortar o parágrafo que começa “A inclinação permite estimar a força horizontal...” e termina “...por essa relação isoladamente.” Permanecem as três limitações e a explicação específica de ida e volta.

8. **Linhas 229–236 — hipótese antes da fórmula.** A projeção central aparece após a equação de escala. Unir a abertura e a explicação: “Para uma câmera vertical sobre terreno plano, o modelo de projeção central representa os raios como passando por um único centro de projeção. Por semelhança de triângulos, a escala é $f_c/h$ [manter a citação]. Aplicada ao espaçamento $p$ entre pixels do sensor, essa escala dá $p/GSD=f_c/h$, em que $GSD$ é a distância de amostragem no solo [manter a citação]:”. Após a equação, manter definições, pixels quadrados, distorção e distinção entre amostragem e resolução efetiva.

9. **Linhas 334–338 e 351–357 — repetição de sinais.** A explicação inicial e os exemplos já justificam os sinais; a linha 357 os justifica pela terceira vez. Cortar apenas “O sinal negativo em $x_c$ faz com que um ponto acima do ponto principal, com $v<c_y$, tenha deslocamento positivo para a frente.” Manter as equações retomadas, as projeções com seno e cosseno e os exemplos A e B.

10. **Linha 409 — parágrafo sobrecarregado.** A soma, a figura, as definições geográficas, o exemplo e as fontes de erro ocupam um bloco. Dividir após “...medido a partir da câmera.” e novamente antes de “Por exemplo...”. Reduzir a descrição intermediária para: “A Figura mostra a origem geográfica e o plano local norte–leste; nele, parte-se da origem, chega-se à aeronave e soma-se o deslocamento até o objeto.” Manter a referência à figura, o exemplo e a ressalva; sua legenda explica o globo e o elipsoide.

11. **Linhas 420–435 — ordem da matriz.** A matriz completa antecede a interpretação da sequência. Mover o parágrafo da linha 435 para imediatamente antes de “Aplicando à câmera...” (linha 422), mantendo matriz, convenções de sinal e explicação das colunas. Depois da matriz, manter a definição dos símbolos e a distinção entre ângulos da câmera e arfagem registrada do gimbal. A redução para visada vertical continua após esses elementos.

12. **Linhas 484–488 — rastreamento e avaliação.** As associações para formar focos e para avaliá-los podem ser confundidas. Iniciar a linha 488 com: “Formar focos previstos e avaliá-los são etapas distintas. Depois da associação temporal e espacial, cada foco previsto é comparado aos objetos anotados por um pareamento um a um.” Prosseguir com classe, distância, cardinalidade e desempate.

## Ajustes pontuais e transições

13. **Linhas 267–276 — alcance da taxa $Vw$.** A área nova contínua e o caso de lacunas entre quadros merecem uma ponte. Trocar a primeira frase da linha 267 por: “Nessas condições e sem revisitas, a faixa varrida continuamente pelo campo de visão cresce à taxa $Vw$. Para que essa taxa também represente terreno registrado, as capturas não podem deixar lacunas.” Preservar a derivação da fração nova e o caso de deslocamento maior que $L$.

14. **Linhas 525–528 — consumo atribuído à inferência.** Seleção e associação também podem variar entre estratégias; a expressão pode sugerir medição isolada da rede. Acrescentar à definição de $E_{\mathrm{inf}}$: “Neste ajuste, ela representa um custo incremental médio associado à carga de passadas, não necessariamente a energia isolada da rede.” Manter a verificação do ajuste e a medição dos componentes completos.

15. **Linha 589 — transição ao Capítulo 3.** A introdução informa que o Capítulo 3 caracteriza a base e ainda receberá os procedimentos experimentais. Evitar prometer que todos os termos já são medidos nos capítulos seguintes. Substituir a última frase por: “O Capítulo 3 caracteriza a base MBG-V2 e os dados disponíveis para estimar parte desses termos; os procedimentos experimentais completarão a avaliação da detecção e da computação embarcada.”
