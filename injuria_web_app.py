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

#Orientando visão
st.markdown('*__Observação: para mais informações acerca do projeto, clique na seta no canto esquerdo superior da tela__*')
st.markdown(' ')

#Informações em sidebar
foto = Image.open('brn.png')
st.sidebar.image(foto, use_column_width=True)
st.sidebar.subheader('Bruno Rodrigues Carloto')
st.sidebar.markdown('Cientista de dados')
st.sidebar.markdown('#### Projeto de portfólio de Ciência de Dados')
st.sidebar.markdown('Em breve haverá artigo')

#Criação de páginas
st.sidebar.title('Menu')
pag = st.sidebar.selectbox('Selecione a página:', ['Experimentar o modelo', 'Sobre o modelo'])
st.markdown(' ')

st.sidebar.markdown("Redes Sociais :")
st.sidebar.markdown("- [Linkedin](https://www.linkedin.com/in/bruno-rodrigues-carloto)")
st.sidebar.markdown("- [Medium](https://br-cienciadedados.medium.com)")
st.sidebar.markdown("- [Github](https://github.com/brunnosjob)")


#Desenvolvimento das páginas
if pag == 'Experimentar o modelo':
    #Cabeçalho
    st.subheader('Detecção de crime de injúria racial e/ou de racismo')
    st.write('''
               O presente modelo de machine learning serve para detecção de crimes de injúria racial e/ou de racismo cometidos em comentários de rede social.
               O teste do modelo não compromete sua pessoa. Você pode simular discursos racistas ou não racistas para experimentar a eficiência do modelo.
               ''')
    #Inserção da frase
    st.subheader('Teste o modelo com diferentes discursos')
    DISCURSO = st.text_input('Faça comentários como se estivesse em uma publicação de rede social:')

    #Definição de dicionário
    palavra_posicao = {'ferrou': 0, 'foi': 1, 'acredito': 2, 'pensei': 3, 'fofo': 4, 'burros': 5, 'gentil': 6, 'perfeita': 7, 'ser': 8, 'bolsonaristas': 9, 'prefiro': 10, 
                            'volta': 11, 'tinha': 12, 'burro': 13, 'dura': 14, 'amor': 15, 'no': 16, 'chineses': 17, 'triste': 18, 'sua': 19, 'isso': 20, 
                            'idiota': 21, 'sai': 22, 'ele': 23, 'porcos': 24, 'coitado': 25, 'legal': 26, 'pássaro': 27, 'até': 28, 'jamais': 29, 'besta': 30,
                           'deveriam': 31, 'muito': 32, 'deles': 33, 'leal': 34, 'deve': 35, 'índia': 36, 'deixa': 37, 'da': 38, 'lamentável': 39, 'sexy': 40,
                           'uau': 41, 'aos': 42, 'veio': 43, 'dele': 44, 'ela': 45, 'baitola': 46, 'são': 47, 'conversei': 48, 'não': 49, 'europeu': 50,
                           'liberte-se': 51, 'esquerdistas': 52, 'branca': 53, 'morressem': 54, 'quer': 55, 'hitler': 56, 'fogo': 57, 'superiores': 58, 'ximpanzé': 59, 'imbecil': 60, 
                           'prevalecer': 61, 'mate': 62, 'outro': 63, 'embora': 64, 'o': 65, 'falo': 66, 'só': 67, 'essa': 68, 'povo': 69, 'petistas': 70,
                           'patamar': 71, 'cara': 72, 'liberte': 73, 'casarei': 74, 'falou': 75, 'complexo': 76, 'aí': 77, 'cai': 78, 'maldição': 79, 'vacilão': 80, 
                           'eu': 81, 'de': 82, 'árabes': 83, 'coisa': 84, 'caso': 85, 'devem': 86, 'verme': 87, 'expulsar': 88, 'faz': 89, 'as': 90, 
                           'mulher': 91, 'menina': 92, 'escravidão': 93, 'índio': 94, 'judeus': 95, 'japonesa': 96, 'africano': 97, 'brasil': 98, 'detesto': 99, 'nível': 100, 
                           'humano': 101, 'inteligentes': 102, 'vem': 103, 'tem': 104, 'esse': 105, 'crentes': 106, 'morra': 107, 'puta': 108, 'uma': 109, 'doente': 110, 
                           'longe': 111, 'manda': 112, 'correr': 113, 'fosse': 114, 'gente': 115, 'nojenta': 116, 'japonês': 117, 'caminho': 118, 'desgraça': 119, 'melhores': 120,
                           'traidor': 121, 'matems': 122, 'fome': 123, 'vida': 124, 'palhaço': 125, 'fossem': 126, 'lindos': 127, 'sociedade': 128, 'escravizado': 129, 'pilantra': 130, 
                           'latinoamericano': 131, 'natureza': 132, 'nadar': 133, 'gostei': 134, 'inferno': 135, 'mataria': 136, 'latinoamericanos': 137, 'é': 138, 'mim': 139, 'bons': 140,
                           'africanos': 141, 'fedorento': 142, 'perfeitas': 143, 'feio': 144, 'real': 145, 'fora': 146, 'matem': 147, 'estivesse': 148, 'cortar': 149, 'cabeça': 150, 
                           'zika': 151, 'eita': 152, 'asiáticos': 153, 'nascer': 154, 'tão': 155, 'inferior': 156, 'grande': 157, 'daqui': 158, 'narigudo': 159, 'jogar': 160, 
                           'índios': 161, 'judaica': 162, 'branquelos': 163, 'namoro': 164, 'foda': 165, 'sou': 166, 'delas': 167, 'mundo': 168, 'comigo': 169, 'nunca': 170, 
                           'mandar': 171, 'forte': 172, 'frios': 173, 'desse': 174, 'vivem': 175, 'perfeito': 176, 'vive': 177, 'quem': 178, 'olha': 179, 'disso': 180, 
                           'pessoa': 181, 'loira': 182, 'ladrões': 183, 'malditos': 184, 'presidente': 185, 'que': 186, 'toda': 187, 'feia': 188, 'porco': 189, 'judeu': 190, 
                           'queria': 191, 'preta': 192, 'maldito': 193, 'vidas': 194, 'queimar': 195, 'maluco': 196, 'lindo': 197, 'todas': 198, 'atear': 199, 'europa': 200,
                           'comer': 201, 'abençoados': 202, 'neles': 203, 'covarde': 204, 'paraíso': 205, 'orgulho': 206, 'brancos': 207, 'vai': 208, 'nosso': 209, 'podre': 210, 
                           'senzalado': 211, 'fechados': 212, 'esperto': 213, 'pega': 214, 'liberta': 215, 'amigo': 216, 'alvo': 217, 'pegou': 218, 'cores': 219, 'odeio': 220, 
                           'dessa': 221, 'difícil': 222, 'ideia': 223, 'te': 224, 'crente': 225, 'salvos': 226, 'inteligente': 227, 'bandido': 228, 'para': 229, 'seu': 230,
                           'morrer': 231, 'escravo': 232, 'preto': 233, 'senzala': 234, 'judeos': 235, 'bênção': 236, 'filha': 237, 'mais': 238, 'amar': 239, 'pretos': 240,
                           'macaco': 241, 'pele': 242, 'faça': 243, 'bomba': 244, 'inútil': 245, 'vermelho': 246, 'inferiores': 247, 'está': 248, 'pra': 249, 'nem': 250, 
                           'europeus': 251, 'deveria': 252, 'matar': 253, 'tirem': 254, 'fofa': 255, 'assassinos': 256, 'louca': 257, 'feios': 258, 'merece': 259, 'os': 260, 
                           'longa': 261, 'sorte': 262, 'assassino': 263, 'menino': 264, 'alguém': 265, 'na': 266, 'negros': 267, 'japão': 268, 'por': 269, 'burra': 270, 
                           'pobre': 271, 'anta': 272, 'se': 273, 'viado': 274, 'áfrica': 275, 'qualquer': 276, 'pretas': 277, 'ontem': 278, 'quero': 279, 'devemos': 280,
                           'fracos': 281, 'demais': 282, 'feiosa': 283, 'negras': 284, 'vi': 285, 'alguns': 286, 'branco': 287, 'viver': 288, 'língua': 289, 'voce': 290,
                           'hoje': 291, 'me': 292, 'feioso': 293, 'sujeito': 294, 'seria': 295, 'maravilhosa': 296, 'contrato': 297, 'banana': 298, 'aquele': 299, 'sei': 300, 
                           'macaca': 301, 'miserável': 302, 'existir': 303, 'árabe': 304, 'um': 305, 'importam': 306, 'merecem': 307, 'louco': 308, 'pro': 309, 'a': 310, 
                           'ridículo': 311, 'coração': 312, 'leva': 313, 'ladrão': 314, 'nos': 315, 'ótimo': 316, 'tempo': 317, 'seja': 318, 'antinegro': 319, 'honesto': 320,
                           'tudo': 321, 'calorosos': 322, 'comeu': 323, 'gato': 324, 'mal': 325, 'mercem': 326, 'cafajeste': 327, 'safado': 328, 'do': 329, 'derrubem': 330,
                           'linda': 331, 'esses': 332, 'amo': 333, 'eles': 334, 'ridícula': 335, 'bombril': 336, 'indiano': 337, 'turco': 338, 'gosta': 339, 'cabelo': 340, 
                           'aqui': 341, 'besteira': 342, 'sincera': 343, 'americano': 344, 'neve': 345, 'perfeitos': 346, 'filho': 347, 'parece': 348, 'você': 349, 'atividade': 350,
                           'nojento': 351, 'antijudeu': 352, 'gosto': 353, 'maldoso': 354, 'negra': 355, 'vou': 356, 'dormindo': 357, 'macumbeiro': 358, 'morram': 359, 'negro': 360,
                           'morte': 361, 'todo': 362, 'arrombado': 363, 'desgraçado': 364, 'com': 365, 'preguiçoso': 366}

    #Função para vetorização de discursos
    def vetorizacao(texto):
      vetor = [0] * 367
      for token in texto.split():
          token = token.lower()
          if token in palavra_posicao:
              posicao = palavra_posicao[token]
              vetor[posicao] += 1
      return vetor
         
    #Importação do modelo
    with open('identificador_logistico.pkl', 'rb') as file:
        modelo = pickle.load(file)
         
    #Programa para classificação do discurso
    vetor = vetorizacao(DISCURSO)
    vetor = np.array([vetor])
    classificacao = modelo.predict(vetor)
    if classificacao == 1:
        st.write("A fala '{}' É UM CRIME de injúria racial ou racismo.".format(DISCURSO))
    elif classificacao == 0:
        st.write("A fala '{}' NÃO é um crime de injúria racial ou racismo.".format(DISCURSO))
                           
elif pag == 'Sobre o modelo':
         
      st.subheader('Sobre o modelo')
      st.markdown('''
      As redes sociais, como Instagram, Twiter e Facebook, se mostram um espaço em que o discurso de ódio é encorajado pela distância.
      Contudo, o fato é que, embora haja distância, o criminoso deve ser justamente punido.

      Sendo assim, desenvolvi esse modelo de machine learning (software inteligente artificialmente), o qual tem como objetivo identificar crimes de injúria racial e/ou de racismo em comentários de redes sociais.
         
      Para desenvolvimento do modelo, utilizei técnicas de Ciência de Dados e Processamento de Linguagem Natural.
         
      A presente aplicação web (interface em que está o modelo de machine learning) serve para interação pública, como experiência pessoal. 
      A partir da experimentação, o público pode compreender a utilidade de um modelo de machine learning como auxiliador para
      identificação de crimes de injúria racial e/ou de racismo em comentários de redes sociais.
      
      #### Sobre o treinamento do modelo
         
      Como produto de machine learning, o modelo foi desenvolvido a partir de supervisão, ou seja, diferentes discursos, racistas e não racistas, foram passados ao modelo, para 
      treinamento, e indicados como sendo ou não sendo racistas. Isso é semelhante a ensinar a uma criança o que é um cachorro, usando a técnica de apontar para o animal e dizer qual 
      animal é. Quanto mais a criança ver diferentes tipos de cachorro e os demais animais, mais ela conseguirá identificar os cachorros, com suas particularidades de cor, raça e tamanho, 
      além de compreender o que não é um cachorro.
         
      As limitações do modelo residem em seu aprendizado. A língua, enquanto linguagem humana, é um fenômeno extremamente versátil e complexo. Há uma diversidade de modos de se falar a mesma coisa, 
      sendo assim, o modelo não viu todas as possibilidades de discurso racista e não racista. Portanto, há discursos racistas que podem não ser identificados, assim como, podem haver 
      discursos não racistas entendidos pelo modelo como racistas.
         
      Para o desenvolvimento de um modelo mais preciso na identificação, o trabalho requer um maior corpus (conjunto de dados textuais/linguísticos), com diversidade de discurso e com dados de qualidade, 
      condizentes ao desenvolvimento do modelo.
         ''')
