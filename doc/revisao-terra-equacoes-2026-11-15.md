# Auditoria independente das equações — 15/11/2026

Escopo: inventário das equações em ambientes matemáticos de `doc/**/*.tex` aponta para os capítulos 2 e 3; examinei também suas definições e hipóteses próximas e trechos de experimentos. Revisão algébrica e dimensional independente, sem reexecutar os dados originais ou verificar sistematicamente todas as referências em seus PDFs. Consultei como contexto `doc/auditoria-formulas-verificacoes.txt`, `doc/auditoria-formulas-geometria.txt` e `doc/auditoria-glauert-confirmacao.txt`; suas verificações de fontes não substituem conferência própria. As linhas abaixo referem-se aos arquivos atuais. Valores experimentais e fidelidade dos códigos às descrições não foram auditados integralmente.

## Achados confirmados

1. **Dependência omitida do hardware no recall e na taxa (cap. 2, linhas 513 e 563–583, `eq_coprojeto`, `eq_otimizacao`).** O próprio texto afirma que a sequência avaliada depende de filas e descarte e que capacidade depende do hardware e modo. Contudo escreve `R_f(h,V,pi)` e `r(pi,h,V)`, excluindo H, ao mesmo tempo que varia H e considera descarte. Assim a restrição pode atribuir a dois hardwares o mesmo recall, embora processem quadros diferentes; a taxa solicitada e a efetivamente executada também ficam indistintas. Corrigir com `r_req(pi,h,V)` na restrição de capacidade e `r_exec(pi,h,V,H)` no consumo medido; escrever `R_f(h,V,pi,H)` quando avaliado sobre a sequência efetiva, ou declarar que as configurações admitidas não descartam e justificar essa independência. Especificar se a restrição usa taxa média, pico ou orçamento com margens para latência.

## Ressalvas e verificações necessárias

- **Busca de GSD (cap. 3, linhas 124–135).** O argmin é sobre os pontos efetivamente testados, não garante mínimo global da grade fina integral; a busca grosseira seguida de refinamento local pode perder outro mínimo estreito. Registrar `S` como união de pontos realmente avaliados, fronteiras, regra de desempate e perfil de sensibilidade; testar grade fina integral ou justificar a restrição. A mediana coordenada não é invariante por rotação, já reconhecido na nota.
- **Identificabilidade da escala (cap. 3, linhas 83–103, 157–161).** A igualdade da projeção pressupõe alvo fixo, mesmo ponto físico da caixa, GSD real constante e câmera/telemetria corretas. Erros de sincronização, direção e posição, paralaxe de alvos elevados e distorção podem enviesar o ajuste; dispersão baixa não prova acurácia absoluta. O manuscrito já alerta para consistência interna e calibração por anotações. Recomendo declarar também invariância temporal do GSD na dedução e avaliar transferência sem anotações de teste.
- **Estimativa de área e energia por área (cap. 2, linhas 207–216; cap. 3, linhas 245–247 e 263–267).** `Vw` é taxa de área nova somente para faixa alinhada, sem lacunas/revisitas e em regime ideal; `E/A_img` é métrica agregada de união geométrica, não `P/(Vw)` pontual. A diferença já é em grande parte explicitada. Para reproduzir e comparar, documentar sincronização, amostragem das pegadas, tratamento de lacunas e fronteiras temporais comuns de energia e área. O próprio texto marca pendência de confirmação do processamento de telemetria (cap. 3, linha 294).
- **Modelo de propulsão (cap. 2, linhas 69–104, 120–144).** A expressão de Glauert e o balanço de forças são coerentes com a convenção declarada, mas os parâmetros e a separação induzida/perfil/parasita são aproximações, com interação de rotores e força no plano omitidas. Na solução exigir consistência entre `alpha_d` da equação induzida e inclinação de equilíbrio, ou explicitar defasagem entre disco e corpo. A condição `V >> vh` também depende do ângulo permanecer limitado. Sem medição da curva, não é possível verificar ótimo ou efeitos reais de massa.
- **Modelo afim de computação (cap. 2, linhas 520–539).** `P_c=P_c0+r E_inf` tem unidades corretas e validade restrita à configuração e faixa calibradas. A memória ideal `r≈n_v* Vw/A_t` mede passadas por área somente se se cobrirem áreas úteis sem duplicações/desperdício; suporte de carga, pré/pós-processamento e interferência de CPU/GPU precisam entrar na medição, como o próprio texto prevê. Em comparações de H, garantir mesmo trabalho efetivamente processado.
- **AP/IoU (cap. 2, linhas 464–480).** Equações corretas com uniões e denominadores positivos. Definir comportamento para `VP+FP=0` e `VP+FN=0` também nas métricas por quadro; documentar implementação e tratamento de limiar de IoU exatamente igual a 0,5. Não pressupor equivalência entre AP de implementações distintas.
- **Estimativa de georreferenciamento (cap. 2, linhas 328–445).** Equação e sinais correspondem à convenção escolhida, mas nadir não determina orientação do topo, e guinada da aeronave pode diferir da câmera. Validar convenção dos dados, ponto principal, lever arm, sincronismo e altura do alvo. A posição do centro da caixa é proxy do alvo e pode variar; referência projetada com mesmos erros não valida erro absoluto. `H_e` é altura elipsoidal de referência, não altura acima do solo, como já explicado.
- **Recortes e taxa de quadros (cap. 2, linhas 455–459 e 532–541).** 24 recortes está correto para bordas preenchidas ou última fileira deslocada; se a implementação utilizar passo fixo com sobreposição em toda a grade, recalcular a contagem. A aproximação de cinco observações é média temporal, não garantia ponto a ponto.

## Equações e exemplos verificados como adequados nas hipóteses declaradas

- Pairado `T=2ρAv_h²`, `P=T v_h`, figura de mérito, soma de N rotores e linearização de massa: coerentes; `1,1^(3/2)-1=15,369%`. Com os parâmetros dados, potência induzida ≈87,91 W, perfil ≈73,76 W, aumento total ≈8,36%, com rotação e demais parâmetros constantes. A potência mecânica parasita só aparece uma vez.
- Equilíbrio, `f_e`, equação vetorial e passagem à potência elétrica: álgebra e unidades coerentes para os eixos e hipóteses declarados. Energia `∫UI dt`, trapézios e conversão `J/m² × 10⁶/3600` também.
- Autonomia, alcance, tangência de `P_el/V` e sua forma com cargas fixas: derivadas corretas; comparar extremos além dos estacionários, como já indicado.
- `GSD=hp/f_c`, dimensões de pegada, tempo `L/V`, `n_v≈f_q L/V`, fração nova saturada, relações energia–altura e tamanho, altura limite e borrão: corretos como modelos condicionais. Para 1,4 cm/pixel: 53,76 × 30,24 m, 7,56 s, 226,8 quadros; objeto de 0,4 m ocupa 28,57 e 14,29 pixels a 40 e 80 m. `A_t=80,2816 m²`.
- Transformação geográfica: com guinada 0, topo→norte e direita→leste; com 90°, topo→leste e direita→sul. Para `(x_c,y_c)=(0,7,1,4)` e 45°, `ΔN≈−0,495 m`, `ΔE≈1,485 m`. Matriz completa é o produto padrão `Rz Ry Rx`; raios de curvatura e diferencial local estão consistentes com radianos e altura referencial.
- GSD testado zero: a projeção coincide com a aeronave, não com o objeto. Para passagem ideal completa, reta e uniforme, a mediana da distância ao centro é `L/4`; ela não é a distância entre extremos nem implica que toda trajetória real seja uniforme. A dedução do deslocamento `(1−GSD/GSD_real) Δx_aer` e a distinção entre 2,79 m extremos e 0,55 m mediana estão corretas.
- IoU, precisão, recall, AP11, pareamento um a um por cardinalidade máxima e distância, utilização de EDF/RM, potência afim e análise dimensional do co-projeto: coerentes sob condições descritas. O tratamento de recall indefinido sem focos reais e de precisão indefinida sem previsões está correto.

## Limites

A leitura avaliou todas as equações apresentadas em ambientes `equation` e as relações diretamente associadas; não constitui verificação independente de todos os números empíricos, scripts, gráficos, PDFs citados ou compilação. As citações e convenções precisam ser confrontadas diretamente com as fontes primárias e com implementação/dados para atestar reprodução. Não se diagnosticou erro de sinal em georreferenciamento nem erro algébrico em `GSD=0` ou na dispersão mediana.


## Complemento: proposta mínima para demanda, execução e descarte

A condição atual `r(pi,h,V) <= r_max(H)` (cap. 2, linhas 577–583) restringe uma taxa que, nas linhas 532–539, é determinada pela estratégia antes de considerar o hardware. Lida como demanda, ela excluiria casos em que descarte mantém a execução dentro da capacidade, embora a linha 513 preveja descarte e avaliação na sequência efetiva. Lida como execução, contradiz a ausência de H e não distingue a carga solicitada. Ainda, a linha 528 prevê ajustar o modelo afim para cada estratégia e modo; a dependência `E_inf(H)` na linha 569 não registra essa variação. A proposta abaixo não presume que a energia por passada seja igual entre estratégias e não impõe descarte zero.

**Texto de ligação sugerido:** Seja `r_dem(pi,h,V)` a taxa média de passadas solicitadas pela estratégia antes das limitações de processamento e `r_exec(pi,h,V,H)` a taxa média efetivamente concluída sob a regra de filas e descarte escolhida. A segunda será medida junto à sequência processada; não se presume que seja o mínimo entre demanda e capacidade. Seja `E_inf(H,pi)` a energia incremental média por passada estimada para a estratégia e modo, na faixa de carga correspondente, incluindo os custos adicionais atribuídos ao processamento completo. Quando um ajuste afim não representar adequadamente a potência, substituir essa expressão por potência média total medida para a configuração. Custos de seleção e descarte que persistam mesmo sem uma passada concluída também devem ser incluídos no ajuste ou na potência medida, sem dupla contagem.

```latex
\begin{equation}
  P_c(H,\pi) = P_{c0}(H) +
    r_{\mathrm{exec}}(\pi,h,V,H)\,E_{\mathrm{inf}}(H,\pi).
  \label{eq_pc}
\end{equation}
```

```latex
\begin{equation}
 e(h,V,\pi,H)=\frac{
 P(V;m_0+m_H)/\eta_m+P_{av}+P_{c0}(H)
 +r_{\mathrm{exec}}(\pi,h,V,H)E_{\mathrm{inf}}(H,\pi)
 }{Vw(h)}.
 \label{eq_coprojeto}
\end{equation}
```

```latex
\begin{equation}
 \min_{(h,V,\pi,H)\in\mathcal D} e(h,V,\pi,H)
 \quad\text{sujeito a}\quad
 \mathcal R_f(h,V,\pi,H)\geq\mathcal R_{\min},
 \quad r_{\mathrm{exec}}(\pi,h,V,H)\leq r_{\max}(H,\pi).
 \label{eq_otimizacao}
\end{equation}
```

O limite de capacidade deve ser medido para o processamento completo e para o mesmo detector, estratégia e modo; a desigualdade de taxas médias é apenas uma condição agregada, não certifica prazos nem estabilidade de fila. Definir o conjunto admissível com a regra concreta de descarte, limites de latência ou fila que forem de fato adotados, e medir taxa, energia e recall na mesma sequência executada. A condição `r_exec <= r_max` pode ser praticamente tautológica se `r_max` for capacidade observada com a mesma política: sua função é explicitar o limite agregado, não substituir medições de latência ou estabelecer viabilidade por si só. O recall sobre todos os objetos da área comum penaliza os não observados devido ao descarte, sem vetar descarte automaticamente; informar também precisão e falsos positivos.

As expressões das linhas 534–539 e as curvas da Figura `fig_passadas` (linhas 549–555, Figura 8 conforme numeração indicada) devem ser interpretadas como **demanda ideal de passadas por área**, `10^6 r_dem/(Vw)`, antes de filas, atrasos e descarte. São médias de regime ideal, não passadas concluídas medidas, nem comprovação de capacidade, latência, qualidade ou custo efetivo. Apresentar separadamente `10^6 r_exec/(Vw)` medido, fração descartada e recall para comparação operacional. A identidade de parcela computacional constante por área da linha 547 também fica condicionada a execução da demanda ideal e ajuste afim válido.

### Reclassificação da fronteira energética

Retiro o item 2 dos achados confirmados: o manuscrito já declara que `P_av` exclui o computador (cap. 2, linha 172), distingue resultados medidos de extrapolações (cap. 2, linha 587), qualifica o consumo de voo e o denominador de área (cap. 3, linhas 262–267) e trata a telemetria como referência pontual, sem identificar a curva (cap. 3, linha 273). Portanto, a diferença de fronteiras é uma **ressalva de aplicação**, não erro confirmado. Caso se combine quantitativamente a telemetria com o cenário hipotético, documentar os componentes efetivamente instalados e a fronteira usada para acrescentar massa e consumo sem dupla contagem; a identidade exata desses componentes não foi confirmada nesta auditoria.

## Alternativa preferida: preservar a restrição de demanda original

A proposta do coordenador é logicamente coerente e preserva a intenção do protocolo: usar `r(π,h,V)` como demanda média ideal, manter `r ≤ r_max(H,π)` como condição de capacidade média e usar `r_exec(π,h,V,H)` na energia. Ela não equivale a exigir ausência de descarte: ainda podem ocorrer picos, atrasos e descarte mesmo com demanda média abaixo do limite. O recall `R_f(h,V,π,H)` deve ser calculado sobre a sequência efetivamente processada, incluindo no denominador os alvos da área comum. Registrar latência, filas e descartes, pois a desigualdade média não certifica prazos. `r_exec=r` quando todas as passadas solicitadas são concluídas no intervalo de avaliação; em janela finita, registrar também trabalhos ainda em fila para não confundir pendência com descarte.

A notação `r_max(H,π)` torna explícita a calibração por estratégia, desde que o limite seja medido com o processamento completo e regras comparáveis. Ao manter o mesmo limite que se pretendia originalmente para cada configuração, a proposta não amplia o conjunto admissível; se anteriormente `r_max(H)` era de fato um único número comum, substituí-lo por limites medidos específicos poderá alterar numericamente o conjunto, embora não altere o tipo de restrição.

```latex
\begin{equation}
 P_c(H,\pi,h,V)=P_{c0}(H)+
 r_{\mathrm{exec}}(\pi,h,V,H)\,E_{\mathrm{inf}}(H,\pi).
 \label{eq_pc}
\end{equation}
```

```latex
\begin{equation}
 e(h,V,\pi,H)=\frac{P(V;m_0+m_H)/\eta_m+P_{av}+P_{c0}(H)
 +r_{\mathrm{exec}}(\pi,h,V,H)E_{\mathrm{inf}}(H,\pi)}{Vw(h)}.
 \label{eq_coprojeto}
\end{equation}
```

```latex
\begin{equation}
 \min_{(h,V,\pi,H)\in\mathcal D} e(h,V,\pi,H)
 \quad\text{sujeito a}\quad
 \mathcal R_f(h,V,\pi,H)\geq\mathcal R_{\min},
 \quad r(\pi,h,V)\leq r_{\max}(H,\pi).
 \label{eq_otimizacao}
\end{equation}
```

A Figura `fig_passadas` representa `10^6 r/(Vw)` como demanda ideal, não passadas concluídas nem demonstração de capacidade ou desempenho. As identidades energéticas das linhas 543–547 e a igualdade da parcela de inferência a `n_v^* E_inf/A_t` requerem `r_exec=r`, hipóteses geométricas ideais e ajuste afim válido. O custo de seleção, descarte e processamento dos trabalhos não concluídos deve estar contemplado no ajuste de potência, ou substitui-se o ajuste por potência total medida. Essa alternativa substitui a restrição `r_exec ≤ r_max` proposta no complemento anterior, que poderia ser tautológica.
