import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataframe = pd.read_excel("./tabela_multivariavel_energia_5000.xlsx", sheet_name="Dados")
x1_trein = dataframe["Temperatura (°C)"].values
x2_trein = dataframe["Umidade (%)"].values
x3_trein = dataframe["Ocupação (pessoas)"].values
x4_trein = dataframe["Hora (0-23)"].values
x5_trein = dataframe["Dia da semana (1-7)"].values
x6_trein = dataframe["Fim de semana (0/1)"].values
y_trein = dataframe["Consumo (kWh)"].values

# escalonamento dos dados dividindo o array de cada x pelo seu valor máximo
x1_trein = x1_trein / np.max(x1_trein)
x2_trein = x2_trein / np.max(x2_trein)
x3_trein = x3_trein / np.max(x3_trein)
x4_trein = x4_trein / np.max(x4_trein)
x5_trein = x5_trein / np.max(x5_trein)
x6_trein = x6_trein / np.max(x6_trein)

# função para calcular a regressão linear múltipla: w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + b
def calcular_regressao_linear_multivariada(x1, x2, x3, x4, x5, x6):
    w1 = 1
    w2 = 5
    w3 = 6
    w4 = 6
    w5 = 3
    w6 = -2
    b = 7
    
    num_iterations = 5000
    y_pred = []

    for i in range(num_iterations):
        # Cálculo das previsões
        y_pred.append(float(w1 * x1[i] + w2 * x2[i] + w3 * x3[i] + w4 * x4[i] + w5 * x5[i] + w6 * x6[i] + b))
    return y_pred

# função para calcular o custo J(w, b)
def calcular_custo(y, y_pred):
    m = len(y)
    custo = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
    return custo
        
pred = calcular_regressao_linear_multivariada(x1_trein, x2_trein, x3_trein, x4_trein, x5_trein, x6_trein)

print()
print("Custo J(w, b):", calcular_custo(y_trein, pred))
print()

plt.scatter(y_trein, pred, label='Previsões do Modelo')
plt.plot([min(y_trein), max(y_trein)], [min(y_trein), max(y_trein)], color='red', linestyle='--', label='Previsão Perfeita')
plt.xlabel('Valores Observados (Consumo Real)')
plt.ylabel('Valores Previstos (Consumo Predito)')
plt.title('Observado vs Previsto')
plt.legend()
plt.show()