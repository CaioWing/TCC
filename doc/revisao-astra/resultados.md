# Capítulo 4 — conferência das evidências

## Validação do detector

O máximo da coluna metrics/mAP50-95(B) em runs/mbg_y11n_640_v1/results.csv é da época 17. Os indicadores por classe vêm de runs/mbg_y11n_640_v1_val_e17/por_classe.json. P e R são apresentados em limiar comum, enquanto F1 máximo seleciona o limiar de cada classe. O denominador da validação é a caixa nos recortes retidos. Os 7.944 recortes incluem parte dos vazios.

A dissertação de [Souza, COPPE/UFRJ, 2025](https://w1files.solucaoatrio.net.br/atrio/ufrj-pee_upl/THESIS/10004597/final__coppe_ufrj_dissertation_mila_enhanced_edition__compressed_20250415093915828.pdf) apresenta os resultados do quinto grupo na tabela A.7. O maior valor tabulado é 0,59. A tabela discrimina intervalos de confiança e o critério para conservar pneus cortados. A comparação com o YOLO11n descreve dois protocolos. A comparação indireta baseada na dissertação de Melo foi retirada.

## Localização, tempo e cobertura

Em doc/dados/qualidade_memoria/quadrantes, resumo.csv e altitude/resumo_seis_voos.csv registram 1.924 trilhas, 8.796 recortes por seleção e 1.131.600 na integral. Recall é 1.079/1.924 ou 1.054/1.924. Precisão é 1.079/2.370 ou 1.054/2.780. Na integral são 1.400 acertos e 5.892 alertas sem par. Os tempos até um segundo incluem todas as referências, enquanto os quantis excluem trilhas sem confirmação. O pareamento espacial usa a classe e o centro projetado, conforme code/avaliacao_mbg/metricas.py. As referências são anotações projetadas e não coordenadas levantadas no terreno.

Recalculada a cobertura pelo quociente entre somas das células de area_recortes_por_voo.csv, obtendo 94,486% e 98,167%. Em alertas_por_movimento.csv a linha “todos” consolida seis voos, somada separadamente dos voos individuais. Das 216 trilhas de maior variação, 96+14 ou 96+19 receberam alerta até cinco segundos. Entre 1.685, são 817+153 ou 817+103. As 23 com janela incompleta pertencem ao denominador geral.

Em sobreposicao_passagens/resumo.csv, origem media_do_voo e video total, há 4.042 pares de recortes, 953 pares com trilha em comum e 1.401 oportunidades para 1.312 trilhas distintas. As categorias exclusivas somam 648+136+140+477=1.401. As duas distribuições de caixas na matriz condicionam nos mesmos 648 reconhecimentos duplos. As médias da IoU são 0,17952 e 0,18353. A origem alternativa gera 4.051 pares e 624 reconhecimentos duplos.

## Potência, área e visibilidade

Em doc/dados, sensibilidade_cruzeiro.csv registra 1.103 janelas, 91,92 min, fator 0,836314 e erro absoluto médio de 4,197 W. curva_potencia_por_voo.csv separa ajuste e comparação. Os dois erros a 7,9 m/s são −2,96 e −6,63 W. visibilidade_por_classe.csv soma 8.441 trilhas nas doze classes. A medição da visibilidade e o ajuste do GSD aproveitam anotações dos mesmos voos. O vento não foi medido e há apenas dois regimes de cruzeiro.

## Bancada e energia

data/yolo11n-jetson24-20261116/summary.csv e code/plot_bancada_motores.py distinguem as medições de imagem a detecções das medições com tensor pronto. No TensorRT nativo, FP16 alcançou 44,87228 imagens/s e 45,72335 mJ acima do repouso. As mAP em FP32, FP16 e INT8 foram 0,41192, 0,41238 e 0,35931 no COCO. O ensaio de qualidade tem 500 imagens e a bancada de tempo/energia tem 95. A projeção da fila utiliza fila_jetson_por_voo.csv; 8.796 × 45,72335 mJ equivalem a 0,1117 Wh.

As linhas dos voos 9, 10, 15, 16, 23 e 24 de balanco_telemetria.csv somam 80,0772 Wh medidos sem kit. As projeções vêm de potencia_estrategias_motores.csv, code/plot_energia_por_area.py e code/sensibilidade_energia_area.py. A energia por área é calculada por voo antes da mediana. Em energia_altura_percursos.csv, a mediana passa de 270,314 para 172,255 Wh/km². Na imagem com escala de 80 m, 888 e 884 referências foram localizadas, frente a 1.079 e 1.054 na escala original. A reprojeção espacial amplia o denominador da energia. Nenhuma referência adicional foi observada na faixa ampliada.

## Apresentação e limites da conferência

A fonte da figura conjunta de altura explicita os dois denominadores e a figura de sobreposição distingue casos pareados de trilhas distintas. O log de compilação disponível antes da integração não apresentava caixa excedente na inclusão do mapa. A caixa excedente de 15,80 pt correspondia à URL do JetPack na bibliografia. Esta conferência utiliza produtos de dados depositados e fórmulas dos scripts; não repetiu treinamento, bancada, conversão de precisão ou aquisição em campo.
