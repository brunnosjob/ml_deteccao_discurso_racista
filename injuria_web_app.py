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
    palavra_posicao = {'filho': 0, 'palhaço': 1, 'senzalado': 2, 'menina': 3, 'ela': 4, 'besteira': 5, 'escravidão': 6, 'índio': 7, 'daqui': 8, 'gostei': 9, 'humano': 10, 'preta': 11, 'longa': 12, 'idiota': 13, 'odeiam': 14, 'todas': 15, 'comeu': 16, 'feioso': 17, 'amar': 18, 'presidente': 19, 'ser': 20, 'salvos': 21, 'desgraçado': 22, 'vi': 23, 'isso': 24, 'a': 25, 'deveriam': 26, 'podre': 27, 'prefiro': 28, 'que': 29, 'inútil': 30, 'áfrica': 31, 'é': 32, 'ladrão': 33, 'fedorento': 34, 'negra': 35, 'voce': 36, 'por': 37, 'louca': 38, 'tudo': 39, 'dormindo': 40, 'negros': 41, 'cabeça': 42, 'caso': 43, 'bandido': 44, 'esse': 45, 'complexo': 46, 'vivem': 47, 'inteligente': 48, 'olha': 49, 'de': 50, 'o': 51, 'essa': 52, 'puta': 53, 'escravizado': 54, 'coitado': 55, 'brancos': 56, 'mandar': 57, 'nosso': 58, 'sei': 59, 'leal': 60, 'orgulho': 61, 'até': 62, 'no': 63, 'maldoso': 64, 'perfeito': 65, 'perfeitas': 66, 'feios': 67, 'ridículo': 68, 'merece': 69, 'filha': 70, 'uma': 71, 'todo': 72, 'esses': 73, 'falou': 74, 'morrer': 75, 'índia': 76, 'quem': 77, 'gente': 78, 'superiores': 79, 'ideia': 80, 'mundo': 81, 'besta': 82, 'mim': 83, 'brasil': 84, 'morte': 85, 'senzala': 86, 'estivesse': 87, 'europa': 88, 'inteligentes': 89, 'pra': 90, 'nos': 91, 'foi': 92, 'ontem': 93, 'vacilão': 94, 'narigudo': 95, 'antijudeu': 96, 'grande': 97, 'eles': 98, 'anta': 99, 'filhos': 100, 'da': 101, 'tirem': 102, 'burros': 103, 'nojento': 104, 'ótimo': 105, 'hitler': 106, 'nem': 107, 'russos': 108, 'judias': 109, 'latinoamericano': 110, 'você': 111, 'dessa': 112, 'zika': 113, 'os': 114, 'morra': 115, 'macumbeiro': 116, 'fogo': 117, 'filhas': 118, 'bomba': 119, 'inferno': 120, 'ele': 121, 'derrubem': 122, 'pretos': 123, 'frios': 124, 'amo': 125, 'asiáticos': 126, 'volta': 127, 'leva': 128, 'contrato': 129, 'forte': 130, 'língua': 131, 'vidas': 132, 'esquerdistas': 133, 'macaca': 134, 'real': 135, 'tempo': 136, 'traidor': 137, 'branco': 138, 'do': 139, 'alguns': 140, 'arrombado': 141, 'existir': 142, 'judeus': 143, 'japonês': 144, 'correr': 145, 'perfeitos': 146, 'burra': 147, 'desse': 148, 'porco': 149, 'caminho': 150, 'odeia': 151, 'europeu': 152, 'está': 153, 'deveria': 154, 'loira': 155, 'cabelo': 156, 'cara': 157, 'parece': 158, 'povo': 159, 'latinoamericanos': 160, 'melhores': 161, 'bons': 162, 'esperto': 163, 'matem': 164, 'negras': 165, 'mal': 166, 'delas': 167, 'matems': 168, 'chineses': 169, 'safado': 170, 'fofa': 171, 'crentes': 172, 'imbecil': 173, 'quer': 174, 'tão': 175, 'verme': 176, 'deixa': 177, 'gato': 178, 'foda': 179, 'covarde': 180, 'assassinos': 181, 'alvo': 182, 'coisa': 183, 'paraíso': 184, 'maldição': 185, 'comigo': 186, 'demais': 187, 'baitola': 188, 'liberte-se': 189, 'fechados': 190, 'crente': 191, 'feiosa': 192, 'nadar': 193, 'viver': 194, 'aqui': 195, 'seria': 196, 'louco': 197, 'lamentável': 198, 'porcos': 199, 'patamar': 200, 'na': 201, 'japonesa': 202, 'sujeito': 203, 'prevalecer': 204, 'eu': 205, 'petistas': 206, 'detesto': 207, 'difícil': 208, 'sorte': 209, 'uau': 210, 'deve': 211, 'triste': 212, 'cores': 213, 'nível': 214, 'africanos': 215, 'longe': 216, 'faça': 217, 'preguiçoso': 218, 'malditos': 219, 'ucranianos': 220, 'deles': 221, 'disso': 222, 'banana': 223, 'abençoados': 224, 'viado': 225, 'africano': 226, 'eita': 227, 'liberte': 228, 'calorosos': 229, 'pobre': 230, 'para': 231, 'japão': 232, 'pega': 233, 'inferior': 234, 'vai': 235, 'maravilhosa': 236, 'comer': 237, 'ladrões': 238, 'branca': 239, 'atear': 240, 'mais': 241, 'gentil': 242, 'nunca': 243, 'assassino': 244, 'hoje': 245, 'acredito': 246, 'mulher': 247, 'quero': 248, 'bênção': 249, 'escravo': 250, 'sexy': 251, 'dura': 252, 'fracos': 253, 'lindo': 254, 'só': 255, 'menino': 256, 'aí': 257, 'queria': 258, 'faz': 259, 'legal': 260, 'árabe': 261, 'fora': 262, 'morressem': 263, 'turco': 264, 'conversei': 265, 'toda': 266, 'amigo': 267, 'tinha': 268, 'te': 269, 'mate': 270, 'desgraça': 271, 'fome': 272, 'negro': 273, 'namoro': 274, 'sai': 275, 'casarei': 276, 'sociedade': 277, 'as': 278, 'pele': 279, 'fofo': 280, 'árabes': 281, 'lindos': 282, 'jamais': 283, 'neles': 284, 'matar': 285, 'pensei': 286, 'sou': 287, 'linda': 288, 'pro': 289, 'expulsar': 290, 'embora': 291, 'macaco': 292, 'honesto': 293, 'antinegro': 294, 'gosta': 295, 'mataria': 296, 'dele': 297, 'falo': 298, 'vida': 299, 'vão': 300, 'mercem': 301, 'judeu': 302, 'morram': 303, 'gosto': 304, 'feia': 305, 'cafajeste': 306, 'seu': 307, 'qualquer': 308, 'índios': 309, 'neve': 310, 'são': 311, 'alguém': 312, 'ridícula': 313, 'seja': 314, 'com': 315, 'importam': 316, 'natureza': 317, 'aos': 318, 'vou': 319, 'americano': 320, 'maldito': 321, 'devem': 322, 'inferiores': 323, 'burro': 324, 'jumentos': 325, 'perfeita': 326, 'me': 327, 'atividade': 328, 'maluco': 329, 'vermelho': 330, 'fossem': 331, 'aquele': 332, 'jogar': 333, 'preto': 334, 'feio': 335, 'odeio': 336, 'ferrou': 337, 'nascer': 338, 'bolsonaristas': 339, 'merecem': 340, 'muito': 341, 'liberta': 342, 'miserável': 343, 'pilantra': 344, 'ximpanzé': 345, 'branquelos': 346, 'amor': 347, 'cortar': 348, 'judeos': 349, 'nojenta': 350, 'jumentas': 351, 'um': 352, 'doente': 353, 'devemos': 354, 'pretas': 355, 'vive': 356, 'coração': 357, 'pessoa': 358, 'indiano': 359, 'se': 360, 'pássaro': 361, 'vem': 362, 'bombril': 363, 'pegou': 364, 'sincera': 365, 'sua': 366, 'europeus': 367, 'veio': 368, 'brasileiros': 369, 'tem': 370, 'não': 371, 'outro': 372, 'cai': 373, 'manda': 374, 'judaica': 375, 'queimar': 376, 'fosse': 377}

    #Função para vetorização de discursos
    def vetorizacao(texto):
      vetor = [0] * 378
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
        st.write("A fala '{}' É SUSPEITA DE COMETER CRIME de injúria racial ou racismo.".format(DISCURSO))
    elif classificacao == 0:
        st.write("A fala '{}' NÃO É SUSPEITA DE COMETEER CRIME de injúria racial ou racismo.".format(DISCURSO))
        
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
