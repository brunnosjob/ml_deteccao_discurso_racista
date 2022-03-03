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
    st.markdown('__O modelo está sendo aperfeiçoado__')
    st.subheader('Simulador')
    st.markdown('#### Detecção de crime de injúria racial e/ou de racismo em discurso')
    
    #Inserção da frase
    st.smarkdown('#### Teste o modelo com diferentes discursos')
    st.markdown('''
    __Exemplos__
    
    - *"Negro maldito"*;
    - *"Os judeus devem morrer"*;
    - *"Quero comer sushi"*;
    - *"Negra maravilhosa"*
    
    Use sua criatividade e considere que o modelo somente conhece 379 palavras e em torno de 500 discursos, 
    o que é pouco, tendo em vista as inúmeras possibilidades de se construir um discurso criminoso ou não criminoso.
    ''')
    DISCURSO = st.text_input('Faça comentários como se estivesse em uma publicação de rede social:')

    #Definição de dicionário
    palavra_posicao = {'filho': 0, 'palhaço': 1, 'senzalado': 2, 'menina': 3, 'ela': 4, 'besteira': 5, 'escravidão': 6, 'índio': 7, 'daqui': 8, 'gostei': 9, 'humano': 10, 'preta': 11, 'longa': 12, 'idiota': 13, 'odeiam': 14, 'todas': 15, 'comeu': 16, 'feioso': 17, 'amar': 18, 'presidente': 19, 'ser': 20, 'salvos': 21, 'desgraçado': 22, 'vi': 23, 'isso': 24, 'a': 25, 'deveriam': 26, 'podre': 27, 'prefiro': 28, 'que': 29, 'inútil': 30, 'áfrica': 31, 'é': 32, 'ladrão': 33, 'fedorento': 34, 'negra': 35, 'voce': 36, 'por': 37, 'louca': 38, 'tudo': 39, 'dormindo': 40, 'negros': 41, 'cabeça': 42, 'caso': 43, 'bandido': 44, 'esse': 45, 'complexo': 46, 'vivem': 47, 'inteligente': 48, 'olha': 49, 'de': 50, 'o': 51, 'essa': 52, 'puta': 53, 'escravizado': 54, 'coitado': 55, 'brancos': 56, 'mandar': 57, 'nosso': 58, 'sei': 59, 'leal': 60, 'orgulho': 61, 'até': 62, 'no': 63, 'lindas': 64, 'maldoso': 65, 'perfeito': 66, 'perfeitas': 67, 'feios': 68, 'ridículo': 69, 'merece': 70, 'filha': 71, 'uma': 72, 'todo': 73, 'esses': 74, 'falou': 75, 'morrer': 76, 'índia': 77, 'quem': 78, 'gente': 79, 'superiores': 80, 'ideia': 81, 'mundo': 82, 'besta': 83, 'mim': 84, 'brasil': 85, 'morte': 86, 'senzala': 87, 'estivesse': 88, 'europa': 89, 'inteligentes': 90, 'pra': 91, 'nos': 92, 'foi': 93, 'ontem': 94, 'vacilão': 95, 'narigudo': 96, 'antijudeu': 97, 'grande': 98, 'eles': 99, 'anta': 100, 'filhos': 101, 'da': 102, 'tirem': 103, 'burros': 104, 'nojento': 105, 'ótimo': 106, 'hitler': 107, 'nem': 108, 'russos': 109, 'judias': 110, 'latinoamericano': 111, 'você': 112, 'dessa': 113, 'zika': 114, 'os': 115, 'morra': 116, 'macumbeiro': 117, 'fogo': 118, 'filhas': 119, 'bomba': 120, 'inferno': 121, 'ele': 122, 'derrubem': 123, 'pretos': 124, 'frios': 125, 'amo': 126, 'asiáticos': 127, 'volta': 128, 'leva': 129, 'contrato': 130, 'forte': 131, 'língua': 132, 'vidas': 133, 'esquerdistas': 134, 'macaca': 135, 'real': 136, 'tempo': 137, 'traidor': 138, 'branco': 139, 'do': 140, 'alguns': 141, 'arrombado': 142, 'existir': 143, 'judeus': 144, 'japonês': 145, 'correr': 146, 'perfeitos': 147, 'burra': 148, 'desse': 149, 'porco': 150, 'caminho': 151, 'odeia': 152, 'europeu': 153, 'está': 154, 'deveria': 155, 'loira': 156, 'cabelo': 157, 'cara': 158, 'parece': 159, 'povo': 160, 'latinoamericanos': 161, 'melhores': 162, 'bons': 163, 'esperto': 164, 'matem': 165, 'negras': 166, 'mal': 167, 'delas': 168, 'matems': 169, 'chineses': 170, 'safado': 171, 'fofa': 172, 'crentes': 173, 'imbecil': 174, 'quer': 175, 'tão': 176, 'verme': 177, 'deixa': 178, 'gato': 179, 'foda': 180, 'covarde': 181, 'assassinos': 182, 'alvo': 183, 'coisa': 184, 'paraíso': 185, 'maldição': 186, 'comigo': 187, 'demais': 188, 'baitola': 189, 'liberte-se': 190, 'fechados': 191, 'crente': 192, 'feiosa': 193, 'nadar': 194, 'viver': 195, 'aqui': 196, 'seria': 197, 'louco': 198, 'lamentável': 199, 'porcos': 200, 'patamar': 201, 'na': 202, 'japonesa': 203, 'sujeito': 204, 'prevalecer': 205, 'eu': 206, 'petistas': 207, 'detesto': 208, 'difícil': 209, 'sorte': 210, 'uau': 211, 'deve': 212, 'triste': 213, 'cores': 214, 'nível': 215, 'africanos': 216, 'longe': 217, 'faça': 218, 'preguiçoso': 219, 'malditos': 220, 'ucranianos': 221, 'deles': 222, 'disso': 223, 'banana': 224, 'abençoados': 225, 'viado': 226, 'africano': 227, 'eita': 228, 'liberte': 229, 'calorosos': 230, 'pobre': 231, 'para': 232, 'japão': 233, 'pega': 234, 'inferior': 235, 'vai': 236, 'maravilhosa': 237, 'comer': 238, 'ladrões': 239, 'branca': 240, 'atear': 241, 'mais': 242, 'gentil': 243, 'nunca': 244, 'assassino': 245, 'hoje': 246, 'acredito': 247, 'mulher': 248, 'quero': 249, 'bênção': 250, 'escravo': 251, 'sexy': 252, 'dura': 253, 'fracos': 254, 'lindo': 255, 'só': 256, 'menino': 257, 'aí': 258, 'queria': 259, 'faz': 260, 'legal': 261, 'árabe': 262, 'fora': 263, 'morressem': 264, 'turco': 265, 'conversei': 266, 'toda': 267, 'amigo': 268, 'tinha': 269, 'te': 270, 'mate': 271, 'desgraça': 272, 'fome': 273, 'negro': 274, 'namoro': 275, 'sai': 276, 'casarei': 277, 'sociedade': 278, 'as': 279, 'pele': 280, 'fofo': 281, 'árabes': 282, 'lindos': 283, 'jamais': 284, 'neles': 285, 'matar': 286, 'pensei': 287, 'sou': 288, 'linda': 289, 'pro': 290, 'expulsar': 291, 'embora': 292, 'macaco': 293, 'honesto': 294, 'antinegro': 295, 'gosta': 296, 'mataria': 297, 'dele': 298, 'falo': 299, 'vida': 300, 'vão': 301, 'mercem': 302, 'judeu': 303, 'morram': 304, 'gosto': 305, 'feia': 306, 'cafajeste': 307, 'seu': 308, 'qualquer': 309, 'índios': 310, 'neve': 311, 'são': 312, 'alguém': 313, 'ridícula': 314, 'seja': 315, 'com': 316, 'importam': 317, 'natureza': 318, 'aos': 319, 'vou': 320, 'americano': 321, 'maldito': 322, 'devem': 323, 'inferiores': 324, 'burro': 325, 'jumentos': 326, 'perfeita': 327, 'me': 328, 'atividade': 329, 'maluco': 330, 'vermelho': 331, 'fossem': 332, 'aquele': 333, 'jogar': 334, 'preto': 335, 'feio': 336, 'odeio': 337, 'ferrou': 338, 'nascer': 339, 'bolsonaristas': 340, 'merecem': 341, 'muito': 342, 'liberta': 343, 'miserável': 344, 'pilantra': 345, 'ximpanzé': 346, 'branquelos': 347, 'amor': 348, 'cortar': 349, 'judeos': 350, 'nojenta': 351, 'jumentas': 352, 'um': 353, 'doente': 354, 'devemos': 355, 'pretas': 356, 'vive': 357, 'coração': 358, 'pessoa': 359, 'indiano': 360, 'se': 361, 'pássaro': 362, 'vem': 363, 'bombril': 364, 'pegou': 365, 'sincera': 366, 'sua': 367, 'europeus': 368, 'veio': 369, 'brasileiros': 370, 'tem': 371, 'não': 372, 'outro': 373, 'cai': 374, 'manda': 375, 'judaica': 376, 'queimar': 377, 'fosse': 378}

    #Função para vetorização de discursos
    def vetorizacao(texto):
      vetor = [0] * 379
      for token in texto.split():
          token = token.lower()
          if token in palavra_posicao:
              posicao = palavra_posicao[token]
              vetor[posicao] += 1
      return vetor
         
    #Importação do modelo
    with open('identificador_multinomial.pkl', 'rb') as file:
        modelo = pickle.load(file)
         
    #Programa para classificação do discurso
    vetor = vetorizacao(DISCURSO)
    vetor = np.array([vetor])
    classificacao = modelo.predict(vetor)
    if classificacao == 1:
        st.write("O discurso '{}' VIOLA A LEI de injúria racial ou de racismo.".format(DISCURSO))
    elif classificacao == 0:
        st.write("O discurso '{}' NÃO VIOLA A LEI de injúria racial ou de racismo.".format(DISCURSO))
        
  
                           
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
      sendo assim, o modelo não viu todas as possibilidades de discurso racista e não racista, havendo sido treinado com 379 palavras e cerca de 500 discursos. 
      Portanto, há discursos racistas que podem não ser identificados, assim como, podem haver 
      discursos não racistas entendidos pelo modelo como racistas.
         
      Para o desenvolvimento de um modelo mais preciso na identificação, o trabalho requer um maior corpus (conjunto de dados textuais/linguísticos), com diversidade de discurso e com dados de qualidade, 
      condizentes ao desenvolvimento do modelo.
      
      #### Limitações do presente modelo
      
      O modelo desenvolvido é para projeto de portfólio. Serve como um protótipo para demonstração de conhecimento. Suas limitações residem no corpus usado, o qual apresenta 379 
      palavras distintas e aproximadamente 500 discursos distintos.
      A quantidade de exemplos 
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
