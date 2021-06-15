import random

import numpy as np

class Rede(object):

    def __init__(self, neuronios):
        self.num_camadas = len(neuronios)
        self.neuronios = neuronios
        self.vies = [np.random.randn(y, 1) for y in neuronios[1:]]
        self.pesos = [np.random.randn(y, x) for x, y in zip(neuronios[:-1], neuronios[1:])]

    #Realiza a multiplicação das entradas pelo peso e soma com o viés
    def feedfoward(self, a):
        for p,v in zip(self.vies, self.pesos):
            a = sigmoid(np.dot(p,a)+v)
        return a

    #Função que treina e aperfeçoa os pesos e o vies, para um melhor resultado de saída
    def gradiente(self, dados_treinamento, epocas, mini_lote, aprendizagem, dados_teste=None):
        dados_treinamento = list(dados_treinamento)
        n = len(dados_treinamento)

        if dados_teste:
            dados_teste = list(dados_teste)
            n_test = len(dados_teste)

        for j in range(epocas):
            random.shuffle(dados_treinamento)
            mini_lotes = [dados_treinamento[k:k + mini_lote] for k in range(0, n, mini_lote)]

            for mini_lote in mini_lotes:
                self.update_mini_lote(mini_lote, aprendizagem)

            if dados_teste:
                print(f"Epocas {j}: {self.avaliacao(dados_teste)} / {n_test}")
            else:
                print(f"Epocas {j} finalizada")

    #atualização dos pesos e vies
    def update_mini_lote(self, mini_lote, aprendizagem):
        nabla_v = [np.zeros(v.shape) for v in self.vies]
        nabla_p = [np.zeros(p.shape) for p in self.pesos]

        for x, y in mini_lote:
            delta_nabla_v, delta_nabla_p = self.backprop(x, y)
            nabla_v = [nv+dnv for nv, dnv in zip(nabla_v, delta_nabla_v)]
            nabla_p = [nap + dnp for nap, dnp in zip(nabla_p, delta_nabla_p)]

        self.pesos = [p - (aprendizagem / len(mini_lote)) * np for p, np in zip(self.pesos, nabla_p)]
        self.vies = [v - (aprendizagem / len(mini_lote)) * nv for v, nv in zip(self.vies, nabla_v)]

    #Realiza a
    def backprop(self, x, y):

        nabla_v = [np.zeros(v.shape) for v in self.vies]
        nabla_p = [np.zeros(p.shape) for p in self.pesos]

        ativacao = x

        ativacoes = [x]

        zs = []

        for v, p in zip(self.vies, self.pesos):
            z = np.dot(p, ativacao) + v
            zs.append(z)
            ativacao = sigmoid(z)
            ativacoes.append(ativacao)

        delta = self.derivadas(ativacoes[-1], y) * sigmoid_prime(zs[-1])
        nabla_v[-1] = delta
        nabla_p[-1] = np.dot(delta, ativacoes[-2].transpose())

        for l in range(2, self.num_camadas):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.pesos[-l + 1].transpose(), delta) * sp
            nabla_v[-l] = delta
            nabla_p[-l] = np.dot(delta, ativacoes[-l - 1].transpose())
        return (nabla_v, nabla_p)

    #retorna o numero de entradas de teste
    def avaliacao(self, dados_teste):

        resultados = [(np.argmax(self.calculo(x)), y) for (x, y) in dados_teste]
        return sum(int(x == y) for (x, y) in resultados)

    #derivadas parcias
    def derivadas(self, atv_saidas, y):
        return (atv_saidas - y)

#função de ativação
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

#Retorna as derivadas da função
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
