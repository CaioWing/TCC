# Altura simulada e inferência integral

O ensaio usa os seis voos completos das regiões E, H e L, desde o quadro zero. O controle de 40 m reproduz as contagens e os alertas por trilha do ensaio original. A simulação mantém seus instantes, suas posições de recorte e suas 1.924 referências. Cada estratégia dispõe de 37.776 recortes em cada altura. O motor e os limiares estão em `protocolo.json`.

## O que varia com a altura

Nos cenários de 50, 60, 70 e 80 m, a entrada do detector continua com 640 × 640 pixels. Para cada recorte escolhido na imagem original de 40 m, extrai-se uma janela concêntrica maior, de lado 640 × altura / 40, e reduz-se a janela com interpolação por área. Um recipiente a 80 m ocupa, assim, metade do lado em pixels. O contexto fora do recorte vem do próprio quadro sempre que está disponível. Preenche-se apenas a parte da janela que ultrapassa os limites do vídeo.

Para avaliar as mesmas porções de terreno, as caixas voltam às coordenadas originais. Contam-se as que têm centro na região escolhida pelo método, limitadas à borda do recorte. A comparação reúne a perda de detalhe e a mudança de contexto da entrada. Um voo real a maior altura mostraria também terreno novo e poderia modificar as escolhas da memória. Os vídeos de referência não permitem reconstruir esse terreno nem recuperar detalhes ópticos de voos mais baixos.

## Referência integral

A etapa integral vem após as quatro alturas e usa exclusivamente a resolução original. Seus 1.131.600 pares de quadro e recorte correspondem aos 24 recortes de cada um dos 47.150 quadros. O ensaio verifica e reaproveita as previsões originais disponíveis do mesmo motor e calcula as restantes. Os limiares de associação, as referências e o critério para o primeiro alerta são os mesmos das duas estratégias.

## Acompanhamento e resultados

- `tmp/avaliacao_mbg/altitude_integral_v1/progresso.json` registra a situação, a altura, o voo e o último limite de bloco concluído. Os arquivos `noite-*.log` e `noite-*.err` ficam na mesma pasta.
- `por_voo.csv` e `por_trilha.csv` recebem somente combinações de voo e condição que já concluíram a inferência e a avaliação. `resumo_seis_voos.csv` recebe cada condição quando os seis voos estão completos. O controle a 40 m pode aparecer enquanto as demais alturas estão em processamento.
- Ao final da fila, `resultado.md` reúne as métricas, `desempenho_por_altura.pdf` compara os dois métodos e `perdas_pareadas_por_classe.csv` identifica quantos recipientes perderam ou ganharam confirmação em cada classe.
- `tamanho_recall_por_classe.pdf` e o CSV de mesmo nome cruzam as trilhas avaliadas com `data/mbg_v2/objetos_por_trilha.csv`. Cada recipiente tem o mesmo peso na média da classe. Sua medida é a mediana do lado equivalente das caixas anotadas no vídeo original, multiplicada por 40/h. Para as alturas simuladas, o CSV também informa o lado mediano original entre os recipientes localizados a 40 m que perderam ou mantiveram a localização. Uma célula vazia indica que a comparação não se aplica ou que o grupo está vazio. Esses arquivos podem ser regenerados com `python code/avaliacao_mbg/resumir_altitude_integral.py` a partir da raiz do projeto.

A partir da raiz do projeto, o comando abaixo executa ou retoma a fila com o ambiente Python que contém o detector e o TensorRT. Os blocos de 300 quadros são gravados antes da avaliação. As trilhas e os focos são avaliados ao longo do voo inteiro.

```powershell
.\code\avaliacao_mbg\rodar_altitude_integral.ps1 -Python (Get-Command python.exe).Source
```
