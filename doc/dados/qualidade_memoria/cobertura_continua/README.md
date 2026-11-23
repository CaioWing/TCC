# Memória por cobertura no voo completo

O ensaio percorre do primeiro ao último quadro dos seis vídeos de teste. Cada estratégia recebe a mesma quantidade de recortes em cada bloco de 60 quadros, inclusive no último bloco incompleto. A fase inicial é zero. Em um bloco completo, o controle envia o quadro inteiro duas vezes e a memória envia 16 recortes em três instantes. Isso corresponde a aproximadamente 24 recortes por segundo de vídeo. A prioridade é a área ainda sem vista em uma malha de 0,5 m. O mapa persiste entre todos os blocos do voo. A seleção usa a pose e dispensa as anotações.

O motor TensorRT FP16 executa o YOLO11n treinado na MBG. As duas estratégias reutilizam a mesma previsão quando escolhem o mesmo par de quadro e recorte. A NMS e a associação usam apenas as previsões selecionadas por cada método.

## Leitura dos resultados

- `resumo.csv` reúne as seis missões. O recall por foco conta cada trilha anotada uma vez por vídeo. Um falso alarme é um foco geográfico previsto sem correspondência. A avaliação segue os limiares do estudo por cobertura e estende a associação ao voo inteiro.
- `por_voo.csv` contém as contagens por estratégia, a quantidade de recortes e o tempo desde o quadro zero até o primeiro alerta visual correto de qualquer recipiente.
- `por_trilha.csv` contém todas as referências das seis classes, inclusive as que só aparecem em quadros omitidos. O atraso começa na primeira caixa presente da trilha e termina no primeiro quadro com detecção da classe correta, confiança de pelo menos 0,5 e IoU de pelo menos 0,5. Cada caixa pode corresponder a uma única anotação no quadro. Um campo vazio indica que a trilha terminou sem esse alerta.
- `tempo_ate_alerta.png` mostra a fração de todas as trilhas que já tiveram confirmação visual em cada prazo. As medianas e o percentil 90 usam as trilhas com confirmação. A coluna `mediana_comuns_s` usa somente as trilhas que os dois métodos confirmaram.
- `area_recortes_por_voo.csv` compara as células que apareceram no vídeo com as que tiveram o centro dentro de pelo menos um recorte enviado. Usa a mesma malha de 0,5 m e a mesma escala da seleção. Os totais contam cada célula uma vez por voo.
- `alertas_por_movimento.csv` reúne as confirmações pareadas por variação da velocidade horizontal nos cinco segundos após a primeira aparição. `alertas_discordantes_maior_variacao.csv` identifica os casos de maior variação em que apenas um método confirmou a trilha nesse prazo.

Um foco conta como acerto geográfico quando a posição agregada atende ao critério de associação. A confirmação visual exige uma caixa correta no quadro da anotação. As duas contagens podem diferir. Os tempos usam o relógio do vídeo e expressam a espera pela vista que produziu o alerta. A decodificação, a fila e a emissão a bordo exigem uma medição física separada.

## Reprodução

Na raiz do projeto, com os vídeos, as anotações e o ambiente TensorRT usados na avaliação, execute os comandos abaixo. A primeira etapa grava a máscara e as previsões por voo. Uma execução interrompida retoma no primeiro voo sem previsões completas.

```powershell
python code/avaliacao_mbg/cobertura_continua.py inferir
python code/avaliacao_mbg/cobertura_continua.py avaliar
python code/avaliacao_mbg/area_recortes_voos.py
python code/avaliacao_mbg/alertas_por_movimento.py
```
