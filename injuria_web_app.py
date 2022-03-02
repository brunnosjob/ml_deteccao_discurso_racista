#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importando bibliotecas principais
import numpy as np
import streamlit as st
import pickle

#Importando algoritmo de machine learning
from PIL import Image
from sklearn.linear_model import LogisticRegression

#Informações em sidebar
st.sidebar.subheader('Projeto de portfólio de Ciência de Dados')
st.sidebar.markdown('Em breve haverá artigo')
st.sidebar.markdown(' ')

#Redes sociais
st.sidebar.markdown('Feito por : Bruno Rodrigues Carloto')
st.sidebar.markdown("Redes Sociais :")
st.sidebar.markdown("- [Linkedin](https://www.linkedin.com/in/bruno-rodrigues-carloto)")
st.sidebar.markdown("- [Medium](https://br-cienciadedados.medium.com)")
st.sidebar.markdown("- [Github](https://github.com/brunnosjob)")


#Cabeçalho
st.header('Modelo de machine learning para detecção de crime de injúria racial e/ou de racismo')
st.subheader('Detectando discursos criminosos em comentários de redes sociais')
st.subheader(' ')

#Informe
st.write('''
As redes sociais, como Instagram, se mostram um espaço em que o discurso de ódio é encorajado pela distância.
Contudo, o fato é que, embora haja distância, o criminoso deve ser justamente punido.

Sendo assim, essa aplicação web demonstra a utilidade de um modelo de machine learning para auxiliar com a 
identificação de crimes de injúria racial e/ou de racismo em comentários de redes sociais.

#### Injúria racial

O crime de injúria racial está inserido no capítulo dos crimes contra a honra,
previsto no parágrafo 3º do artigo 140 do Código Penal. O crime é caracterizado quando há ofensa à dignidade de alguém, 
com base em elementos referentes à sua raça, cor, etnia, religião, idade ou deficiência. 
Nesta hipótese, a pena pode ir de 1 a 3 anos de reclusão. Esse crime não se confunde com o crime de racismo.

#### Racismo

Os crimes de racismo estão previstos na Lei 7.716/1989, conhecida como Lei do Racismo.
Essa lei foi elaborada para regulamentar a punição de crimes resultantes de preconceito de raça ou de cor. 
Todavia a Lei nº 9.459/13 acrescentou à referida lei os termos etnia, 
religião e procedência nacional, ampliando a proteção para vários tipos de intolerância.
''')

#Inserção da frase
st.write('Teste o modelo com diferentes discursos')
DISCURSO = st.text_input('Faça comentários como se estivesse em uma publicação de rede social:')

#Importação do modelo
with open('identificador_injuria_racismo_versao_logistico_1.pkl', 'rb') as file:
    vetorizador, modelo = pickle.load(file)
    
#Aplicação do modelo em lógica
#Testando o modelo
DISCURSO = 'O Lula odeia os negros'

#Programa para classificação do discurso
vetor = vetorizador(DISCURSO)
vetor = np.array([vetor])
classificacao = modelo.predict(vetor)
if classificacao == 1:
    print("A fala '{}' É UM CRIME de injúria racial ou racismo.".format(DISCURSO))
else:
    print("A fala '{}' NÃO é um crime de injúria racial ou racismo.".format(DISCURSO))

