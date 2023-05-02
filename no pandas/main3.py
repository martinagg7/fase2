

import csv
import random
import math


# abrir el archivo y leerlo
def load_csv(data_year_data):
    dataset = []
    with open(data_year_data, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# ordena data set quitando los blancos
def summarize(dataset):
    summaries = [(column, min(values), max(values), len(values), sum(values) / len(values)) for column, values in enumerate(zip(*dataset))]
    del summaries[-1]                   
    return summaries 


# da la longitud de cada dataset y cuenta los datos de cada archivo
def split_data(dataset, split_ratio):
    train_size = int(len(dataset) * split_ratio)
    train_set = []
    test_set = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(test_set))
        train_set.append(test_set.pop(index))
    return train_set, test_set


# toma fila de datos y los coeficiente para deevolver la prediccion (suma del producto de la variable por el cociente)
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return yhat

    """
    La función predict toma una fila de datos y los coeficientes del modelo, y devuelve la predicción del modelo para esa fila. 
    La predicción se calcula como la suma del producto de cada variable en la fila por su respectivo coeficiente, más el término constante coef[0].

    """


# es el gradiente descendente estocastico
def coefficients_sgd(train, learning_rate, n_epochs):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epochs):
        sum_error = 0.0
        for row in train:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            sum_error += error ** 2
            coef[0] = coef[0] - learning_rate * error
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] - learning_rate * error * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))
    return coef

    """
    La función coefficients_sgd implementa el algoritmo SGD para ajustar los coeficientes del modelo. La función toma como entrada los datos de 
    entrenamiento, la tasa de aprendizaje (learning_rate) y el número de épocas (n_epochs) para iterar el algoritmo. En cada época, el algoritmo 
    recorre todas las filas del conjunto de entrenamiento, y para cada fila, calcula la predicción del modelo (yhat) utilizando la función predict. 
    
    Luego, calcula el error entre la predicción y el valor real de la variable objetivo (error), y actualiza los coeficientes del modelo utilizando 
    la regla de actualización del gradiente descendente estocástico. Finalmente, la función devuelve los coeficientes ajustados.


    El código también imprime el error de entrenamiento en cada época para monitorear el progreso del entrenamiento.

    """