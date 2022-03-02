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
pag = st.sidebar.selectbox('Selecione a página:', ['Experimentar o modelo', 'Sobre o modelo', 'Sobre os crimes'])
st.markdown(' ')

st.sidebar.markdown("Redes Sociais :")
st.sidebar.markdown("- [Linkedin](https://www.linkedin.com/in/bruno-rodrigues-carloto)")
st.sidebar.markdown("- [Medium](https://br-cienciadedados.medium.com)")
st.sidebar.markdown("- [Github](https://github.com/brunnosjob)")


#Desenvolvimento das páginas
if pag == 'Experimentar o modelo':
    #Cabeçalho
    st.markdown('__O modelo está sendo aperfeiçoamento__')
    st.subheader('Detecção de crime de injúria racial e/ou de racismo em discurso')
    st.write('''
               O presente modelo de machine learning serve para detecção de crimes de injúria racial e/ou de racismo cometidos em comentários de rede social.
               O teste do modelo não compromete sua pessoa. Você pode simular discursos racistas ou não racistas para experimentar a eficiência do modelo.
               ''')
    #Inserção da frase
    st.subheader('Teste o modelo com diferentes discursos')
    st.markdown('''
    __Exemplos__
    
    - *"Negro maldito"*;
    - *"Os judeus devem morrer"*;
    - *"Quero comer sushi"*;
    - *"Negra maravilhosa"*
    
    Use sua criatividade
    ''')
    DISCURSO = st.text_input('Faça comentários como se estivesse em uma publicação de rede social:')

    #Definição de dicionário
    palavra_posicao = {'filho': 0, 'palhaço': 1, 'senzalado': 2, 'menina': 3, 'ela': 4, 'besteira': 5, 'escravidão': 6, 'índio': 7, 'daqui': 8, 'gostei': 9, 'humano': 10, 'preta': 11, 'longa': 12, 'idiota': 13, 'todas': 14, 'comeu': 15, 'feioso': 16, 'amar': 17, 'presidente': 18, 'ser': 19, 'salvos': 20, 'desgraçado': 21, 'vi': 22, 'isso': 23, 'a': 24, 'deveriam': 25, 'podre': 26, 'prefiro': 27, 'que': 28, 'inútil': 29, 'áfrica': 30, 'é': 31, 'ladrão': 32, 'fedorento': 33, 'negra': 34, 'voce': 35, 'por': 36, 'louca': 37, 'tudo': 38, 'dormindo': 39, 'negros': 40, 'cabeça': 41, 'caso': 42, 'bandido': 43, 'esse': 44, 'complexo': 45, 'vivem': 46, 'inteligente': 47, 'olha': 48, 'de': 49, 'o': 50, 'essa': 51, 'puta': 52, 'escravizado': 53, 'coitado': 54, 'brancos': 55, 'mandar': 56, 'nosso': 57, 'sei': 58, 'leal': 59, 'orgulho': 60, 'até': 61, 'no': 62, 'maldoso': 63, 'perfeito': 64, 'perfeitas': 65, 'feios': 66, 'ridículo': 67, 'merece': 68, 'filha': 69, 'uma': 70, 'todo': 71, 'esses': 72, 'falou': 73, 'morrer': 74, 'índia': 75, 'quem': 76, 'gente': 77, 'superiores': 78, 'ideia': 79, 'mundo': 80, 'besta': 81, 'mim': 82, 'brasil': 83, 'morte': 84, 'senzala': 85, 'estivesse': 86, 'europa': 87, 'inteligentes': 88, 'pra': 89, 'nos': 90, 'foi': 91, 'ontem': 92, 'vacilão': 93, 'narigudo': 94, 'antijudeu': 95, 'grande': 96, 'eles': 97, 'anta': 98, 'da': 99, 'tirem': 100, 'burros': 101, 'nojento': 102, 'ótimo': 103, 'hitler': 104, 'nem': 105, 'russos': 106, 'latinoamericano': 107, 'você': 108, 'dessa': 109, 'zika': 110, 'os': 111, 'morra': 112, 'macumbeiro': 113, 'fogo': 114, 'bomba': 115, 'inferno': 116, 'ele': 117, 'derrubem': 118, 'pretos': 119, 'frios': 120, 'amo': 121, 'asiáticos': 122, 'volta': 123, 'leva': 124, 'contrato': 125, 'forte': 126, 'língua': 127, 'vidas': 128, 'esquerdistas': 129, 'macaca': 130, 'real': 131, 'tempo': 132, 'traidor': 133, 'branco': 134, 'do': 135, 'alguns': 136, 'arrombado': 137, 'existir': 138, 'judeus': 139, 'japonês': 140, 'correr': 141, 'perfeitos': 142, 'burra': 143, 'desse': 144, 'porco': 145, 'caminho': 146, 'europeu': 147, 'está': 148, 'deveria': 149, 'loira': 150, 'cabelo': 151, 'cara': 152, 'parece': 153, 'povo': 154, 'latinoamericanos': 155, 'melhores': 156, 'bons': 157, 'esperto': 158, 'matem': 159, 'negras': 160, 'mal': 161, 'delas': 162, 'matems': 163, 'chineses': 164, 'safado': 165, 'fofa': 166, 'crentes': 167, 'imbecil': 168, 'quer': 169, 'tão': 170, 'verme': 171, 'deixa': 172, 'gato': 173, 'foda': 174, 'covarde': 175, 'assassinos': 176, 'alvo': 177, 'coisa': 178, 'paraíso': 179, 'maldição': 180, 'comigo': 181, 'demais': 182, 'baitola': 183, 'liberte-se': 184, 'fechados': 185, 'crente': 186, 'feiosa': 187, 'nadar': 188, 'viver': 189, 'aqui': 190, 'seria': 191, 'louco': 192, 'lamentável': 193, 'porcos': 194, 'patamar': 195, 'na': 196, 'japonesa': 197, 'sujeito': 198, 'prevalecer': 199, 'eu': 200, 'petistas': 201, 'detesto': 202, 'difícil': 203, 'sorte': 204, 'uau': 205, 'deve': 206, 'triste': 207, 'cores': 208, 'nível': 209, 'africanos': 210, 'longe': 211, 'faça': 212, 'preguiçoso': 213, 'malditos': 214, 'ucranianos': 215, 'deles': 216, 'disso': 217, 'banana': 218, 'abençoados': 219, 'viado': 220, 'africano': 221, 'eita': 222, 'liberte': 223, 'calorosos': 224, 'pobre': 225, 'para': 226, 'japão': 227, 'pega': 228, 'inferior': 229, 'vai': 230, 'maravilhosa': 231, 'comer': 232, 'ladrões': 233, 'branca': 234, 'atear': 235, 'mais': 236, 'gentil': 237, 'nunca': 238, 'assassino': 239, 'hoje': 240, 'acredito': 241, 'mulher': 242, 'quero': 243, 'bênção': 244, 'escravo': 245, 'sexy': 246, 'dura': 247, 'fracos': 248, 'lindo': 249, 'só': 250, 'menino': 251, 'aí': 252, 'queria': 253, 'faz': 254, 'legal': 255, 'árabe': 256, 'fora': 257, 'morressem': 258, 'turco': 259, 'conversei': 260, 'toda': 261, 'amigo': 262, 'tinha': 263, 'te': 264, 'mate': 265, 'desgraça': 266, 'fome': 267, 'negro': 268, 'namoro': 269, 'sai': 270, 'casarei': 271, 'sociedade': 272, 'as': 273, 'pele': 274, 'fofo': 275, 'árabes': 276, 'lindos': 277, 'jamais': 278, 'neles': 279, 'matar': 280, 'pensei': 281, 'sou': 282, 'linda': 283, 'pro': 284, 'expulsar': 285, 'embora': 286, 'macaco': 287, 'honesto': 288, 'antinegro': 289, 'gosta': 290, 'mataria': 291, 'dele': 292, 'falo': 293, 'vida': 294, 'vão': 295, 'mercem': 296, 'judeu': 297, 'morram': 298, 'gosto': 299, 'feia': 300, 'cafajeste': 301, 'seu': 302, 'qualquer': 303, 'índios': 304, 'neve': 305, 'são': 306, 'alguém': 307, 'ridícula': 308, 'seja': 309, 'com': 310, 'importam': 311, 'natureza': 312, 'aos': 313, 'vou': 314, 'americano': 315, 'maldito': 316, 'devem': 317, 'inferiores': 318, 'burro': 319, 'perfeita': 320, 'me': 321, 'atividade': 322, 'maluco': 323, 'vermelho': 324, 'fossem': 325, 'aquele': 326, 'jogar': 327, 'preto': 328, 'feio': 329, 'odeio': 330, 'ferrou': 331, 'nascer': 332, 'bolsonaristas': 333, 'merecem': 334, 'muito': 335, 'liberta': 336, 'miserável': 337, 'pilantra': 338, 'ximpanzé': 339, 'branquelos': 340, 'amor': 341, 'cortar': 342, 'judeos': 343, 'nojenta': 344, 'um': 345, 'doente': 346, 'devemos': 347, 'pretas': 348, 'vive': 349, 'coração': 350, 'pessoa': 351, 'indiano': 352, 'se': 353, 'pássaro': 354, 'vem': 355, 'bombril': 356, 'pegou': 357, 'sincera': 358, 'sua': 359, 'europeus': 360, 'veio': 361, 'brasileiros': 362, 'tem': 363, 'não': 364, 'outro': 365, 'cai': 366, 'manda': 367, 'judaica': 368, 'queimar': 369, 'fosse': 370}

    #Função para vetorização de discursos
    def vetorizacao(texto):
      vetor = [0] * 371
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
        
    st.markdown('''
    __Analise seu discurso e compare com a resposta/compreensão do modelo__
    
    #### Observação:
    
    Para compreensão acerca do aprendizado do modelo e suas limitações, navegue para a página __Sobre o modelo__ e leia o tópico __Sobre o treinamento do modelo__.
    ''')
                           
elif pag == 'Sobre o modelo':
         
      st.subheader('Sobre o modelo')
      st.markdown('''
      As redes sociais, como Instagram, Twitter e Facebook, se mostram um espaço em que o discurso de ódio é encorajado pela distância.
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
      
      #### Limitações do presente modelo
      
      O modelo desenvolvido é para projeto de portfólio. Serve como um protótipo para demonstração de conhecimento. Suas limitações residem no corpus usado. A quantidade de exemplos 
      inseridos na aprendizagem do modelo é limitada diante da diversidade de discursos criminosos que se enquadram como crimes de injúria racial ou de racismo.
         ''')
        
elif pag == 'Sobre os crimes':
    
    st.subheader('Sobre os crimes')
    st.markdown('''
    #### Injúria racial
    
    O crime de injúria racial está previsto no Código Penal, no parágrafo 3ª do artigo 140. Caracteriza-se pela ofensa à dignidade ou à integridade do indivíduo,
    com base em elementos referentes à sua raça, cor, etnia, religião, origem ou a condição da pessoa idosa ou portadora de deficiência. A pessoa idosa é caracterizada 
    pela idade igual ou superior a 60 anos, de acordo, com o Estatuto do Idoso, previsto na Lei Federal 10.741/03, artigo 1ª.
    
    #### Racismo
    
    Os crimes de racismo estão previstos na Lei 7.716/1989, conhecida como Lei de Racismo.
    A Lei foi elaborada para regulamentar a punição de crimes resultantes de preconceito de raça ou de cor. 
    Contudo, foram acrescentados à referida lei os termos etnia, religião e procedência nacional, através da Lei nº 9.459/13, 
    ampliando a proteção para vários tipos de intolerância. As penas previstas são mais severas e podem chegar até a 5 anos de reclusão.
    
    #### Diferença entre injúria racial e racismo
    
    O que diferencia os crimes é o direcionamento da conduta. Enquanto a injúria racial é a ofensa direcionada a um indivíduo especifico, 
    o crime de racismo é a ofensa contra uma coletividade, por exemplo, toda uma raça ou etnia.
    
    
    
    
    __Agradecimentos__
    
    Agradecimentos a bacharelanda em Direito, Misma Kelly Marcílio Carloto Rodrigues, que auxilou com a devida exposição das leis.
    ''')
