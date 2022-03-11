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
st.sidebar.markdown("- [Artigo com o passo a passo do desenvolvimento do modelo](https://br-cienciadedados.medium.com/machine-learning-para-detec%C3%A7%C3%A3o-de-discurso-racista-parte-1-1b697673d4bc)")

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
    st.markdown('#### Teste o modelo com diferentes discursos')
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
    palavra_posicao = {'feioso': 0, 'ferrou': 1, 'lindos': 2, 'dos': 3, 'maluco': 4, 'amo': 5, 'bom': 6, 'gosto': 7, 'inteligente': 8, 'ladrões': 9, 'perfeito': 10, 'as': 11, 'todas': 12, 'deles': 13, 'atear': 14, 'com': 15, 'sujeito': 16, 'ser': 17, 'lindas': 18, 'maldito': 19, 'parece': 20, 'aqui': 21, 'essa': 22, 'índia': 23, 'filhos': 24, 'europeu': 25, 'índios': 26, 'alguém': 27, 'vai': 28, 'são': 29, 'existisse': 30, 'longe': 31, 'namoro': 32, 'lindo': 33, 'filha': 34, 'judaica': 35, 'gosta': 36, 'latinoamericanos': 37, 'mulher': 38, 'foi': 39, 'mal': 40, 'língua': 41, 'latinoamericano': 42, 'anta': 43, 'que': 44, 'se': 45, 'sexy': 46, 'só': 47, 'ele': 48, 'narigudo': 49, 'americano': 50, 'menina': 51, 'japão': 52, 'cara': 53, 'lamentável': 54, 'larão': 55, 'uau': 56, 'sou': 57, 'quem': 58, 'porcos': 59, 'sincera': 60, 'burros': 61, 'viado': 62, 'está': 63, 'escravo': 64, 'para': 65, 'safado': 66, 'miserável': 67, 'ladrão': 68, 'judeos': 69, 'sociedade': 70, 'quando': 71, 'humanidade': 72, 'crentes': 73, 'forte': 74, 'embora': 75, 'falou': 76, 'podre': 77, 'nos': 78, 'me': 79, 'ximpanzé': 80, 'liberte-se': 81, 'calorosos': 82, 'fez': 83, 'comer': 84, 'da': 85, 'é': 86, 'um': 87, 'hoje': 88, 'fome': 89, 'eles': 90, 'leva': 91, 'vermelho': 92, 'alvo': 93, 'antijudeu': 94, 'fofa': 95, 'daqui': 96, 'traidor': 97, 'derrubem': 98, 'inferior': 99, 'besteira': 100, 'matems': 101, 'nasceu': 102, 'preguiçoso': 103, 'desse': 104, 'ela': 105, 'ideia': 106, 'em': 107, 'seu': 108, 'pássaro': 109, 'patamar': 110, 'índio': 111, 'assassino': 112, 'malditos': 113, 'morrer': 114, 'real': 115, 'morram': 116, 'cai': 117, 'imbecil': 118, 'dessa': 119, 'senzalado': 120, 'nojento': 121, 'ficar': 122, 'cabelo': 123, 'disso': 124, 'vem': 125, 'inteligentes': 126, 'orgulho': 127, 'voce': 128, 'branco': 129, 'inútil': 130, 'jogar': 131, 'nosso': 132, 'maldoso': 133, 'chineses': 134, 'louco': 135, 'brasil': 136, 'baitola': 137, 'africano': 138, 'negros': 139, 'coração': 140, 'cafajeste': 141, 'branca': 142, 'fracos': 143, 'lembre': 144, 'volta': 145, 'mercem': 146, 'pilantra': 147, 'triste': 148, 'pele': 149, 'existissem': 150, 'perfeitos': 151, 'bênção': 152, 'mataria': 153, 'amar': 154, 'tinha': 155, 'deveria': 156, 'feiosa': 157, 'amor': 158, 'neles': 159, 'burro': 160, 'por': 161, 'sua': 162, 'amigo': 163, 'paraíso': 164, 'leal': 165, 'indiano': 166, 'preto': 167, 'ridícula': 168, 'demais': 169, 'porco': 170, 'contrato': 171, 'cortar': 172, 'lugar': 173, 'aos': 174, 'ir': 175, 'bem': 176, 'morra': 177, 'bombril': 178, 'conversei': 179, 'legal': 180, 'palhaço': 181, 'veio': 182, 'asiáticos': 183, 'esquerdistas': 184, 'zika': 185, 'prefiro': 186, 'russos': 187, 'superiores': 188, 'menino': 189, 'bolsonaristas': 190, 'pessoa': 191, 'dormindo': 192, 'judeu': 193, 'vive': 194, 'tempo': 195, 'deveriam': 196, 'os': 197, 'morte': 198, 'vacilão': 199, 'feios': 200, 'besta': 201, 'branquelos': 202, 'longa': 203, 'africanos': 204, 'crente': 205, 'prevalecer': 206, 'japonês': 207, 'uma': 208, 'pega': 209, 'antinegro': 210, 'odeiam': 211, 'cadeia': 212, 'outro': 213, 'manda': 214, 'sorte': 215, 'judias': 216, 'tirem': 217, 'liberta': 218, 'petistas': 219, 'japonesa': 220, 'loira': 221, 'gentil': 222, 'perfeita': 223, 'vivo': 224, 'cabeça': 225, 'liberte': 226, 'inferno': 227, 'povo': 228, 'europeus': 229, 'maravilhosa': 230, 'europa': 231, 'devem': 232, 'dono': 233, 'todos': 234, 'faça': 235, 'bandido': 236, 'qualquer': 237, 'merecem': 238, 'vão': 239, 'não': 240, 'casarei': 241, 'matar': 242, 'bomba': 243, 'esses': 244, 'atividade': 245, 'pensei': 246, 'honesto': 247, 'ontem': 248, 'deixa': 249, 'pra': 250, 'pobre': 251, 'nível': 252, 'mandar': 253, 'turco': 254, 'ridículo': 255, 'no': 256, 'nunca': 257, 'tudo': 258, 'vi': 259, 'covarde': 260, 'fora': 261, 'fofo': 262, 'grande': 263, 'quer': 264, 'negra': 265, 'entrada': 266, 'você': 267, 'jumentos': 268, 'comeu': 269, 'merece': 270, 'odeio': 271, 'tem': 272, 'certo': 273, 'esse': 274, 'aquele': 275, 'macumbeiro': 276, 'desgraçado': 277, 'até': 278, 'mais': 279, 'macaca': 280, 'o': 281, 'doente': 282, 'foda': 283, 'deve': 284, 'hitler': 285, 'brasileiros': 286, 'queria': 287, 'do': 288, 'áfrica': 289, 'morressem': 290, 'delas': 291, 'gato': 292, 'estivesse': 293, 'nojenta': 294, 'morto': 295, 'seja': 296, 'alguns': 297, 'puta': 298, 'difícil': 299, 'mate': 300, 'muito': 301, 'pretos': 302, 'pro': 303, 'neve': 304, 'seria': 305, 'louca': 306, 'expulsar': 307, 'fogo': 308, 'de': 309, 'aí': 310, 'corrupto': 311, 'idiota': 312, 'existir': 313, 'coisa': 314, 'jumentas': 315, 'filhas': 316, 'abençoados': 317, 'mim': 318, 'inferiores': 319, 'ótimo': 320, 'eu': 321, 'sai': 322, 'fedorento': 323, 'toda': 324, 'vidas': 325, 'também': 326, 'falo': 327, 'esperto': 328, 'faz': 329, 'dura': 330, 'vida': 331, 'humano': 332, 'banana': 333, 'negro': 334, 'feio': 335, 'preta': 336, 'céu': 337, 'parte': 338, 'sei': 339, 'negras': 340, 'perfeitas': 341, 'filho': 342, 'eita': 343, 'pegou': 344, 'coitado': 345, 'árabe': 346, 'escravizado': 347, 'matem': 348, 'linda': 349, 'melhores': 350, 'caga': 351, 'gente': 352, 'queimar': 353, 'presidente': 354, 'senzala': 355, 'todo': 356, 'a': 357, 'feia': 358, 'olha': 359, 'fosse': 360, 'cores': 361, 'brancos': 362, 'na': 363, 'salvos': 364, 'viver': 365, 'detesto': 366, 'nem': 367, 'ucranianos': 368, 'isso': 369, 'burra': 370, 'devemos': 371, 'bons': 372, 'frios': 373, 'maldição': 374, 'árabes': 375, 'nego': 376, 'arrombado': 377, 'judeus': 378, 'fossem': 379, 'quero': 380, 'e': 381, 'comigo': 382, 'gostei': 383, 'caminho': 384, 'fechados': 385, 'nadar': 386, 'importam': 387, 'macaco': 388, 'odeia': 389, 'correr': 390, 'complexo': 391, 'nascer': 392, 'vivem': 393, 'pretas': 394, 'acredito': 395, 'te': 396, 'jamais': 397, 'vou': 398, 'natureza': 399, 'caso': 400, 'saida': 401, 'verme': 402, 'escravidão': 403, 'liberdade': 404, 'mundo': 405, 'dele': 406, 'tão': 407, 'desgraça': 408, 'assassinos': 409}

    #Função para vetorização de discursos
    def vetorizacao(texto):
      vetor = [0] * 410
      for token in texto.split():
          token = token.lower()
          if token in palavra_posicao:
              posicao = palavra_posicao[token]
              vetor[posicao] += 1
      return vetor
         
    #Importação do modelo
    with open('identificador_multinomial_versao_2.pkl', 'rb') as file:
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
