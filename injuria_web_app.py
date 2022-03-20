#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importando bibliotecas principais
import numpy as np
import streamlit as st
import pickle
import nltk
import spacy

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
    palavra_posicao = {'': 0, 'antar': 1, 'dormir': 2, 'amigo': 3, 'doente': 4, 'certo': 5, 'vidas': 6, 'japonesa': 7, 'humanidade': 8, 'alguém': 9, 'burros': 10, 'latinoamericano': 11, 'ucranianos': 12, 'embora': 13, 'comer': 14, 'preta': 15, 'lugar': 16, 'japão': 17, 'maravilhosa': 18, 'ladrões': 19, 'falir': 20, 'idiota': 21, 'cara': 22, 'fofo': 23, 'amor': 24, 'merecer': 25, 'toda': 26, 'nível': 27, 'olhar': 28, 'daqui': 29, 'odeio': 30, 'alguns': 31, 'uau': 32, 'ir': 33, 'matar': 34, 'falar': 35, 'liberte-se': 36, 'dono': 37, 'europeus': 38, 'dura': 39, 'branquelos': 40, 'maldito': 41, 'negra': 42, 'feio': 43, 'bombril': 44, 'leal': 45, 'outro': 46, 'esperto': 47, 'nadar': 48, 'miserável': 49, 'orgulho': 50, 'expulsar': 51, 'difícil': 52, 'pobre': 53, 'traidor': 54, 'vou': 55, 'eita': 56, 'cadeia': 57, 'europa': 58, 'desse': 59, 'turco': 60, 'bons': 61, 'mercem': 62, 'brasileiros': 63, 'tudo': 64, 'filho': 65, 'imbecil': 66, 'inferno': 67, 'banana': 68, 'pensar': 69, 'petistas': 70, 'ótimo': 71, 'parece': 72, 'índios': 73, 'mandar': 74, 'sorte': 75, 'índio': 76, 'ximpanzé': 77, 'lindo': 78, 'feia': 79, 'mal': 80, 'matems': 81, 'bem': 82, 'céu': 83, 'atear': 84, 'inteligentes': 85, 'macumbeiro': 86, 'morto': 87, 'voce': 88, 'judeos': 89, 'libertar': 90, 'honesto': 91, 'levar': 92, 'nojento': 93, 'fracos': 94, 'fedorento': 95, 'feios': 96, 'língua': 97, 'coisa': 98, 'maldição': 99, 'maldoso': 100, 'cortar': 101, 'mim': 102, 'zika': 103, 'judaica': 104, 'vermelho': 105, 'ladrão': 106, 'árabe': 107, 'lindar': 108, 'acreditar': 109, 'bom': 110, 'patamar': 111, 'perfeitos': 112, 'vão': 113, 'perfeito': 114, 'besteira': 115, 'gostar': 116, 'bênção': 117, 'indiano': 118, 'larão': 119, 'malditos': 120, '.': 121, 'macaco': 122, 'asiáticos': 123, 'baitola': 124, 'ficar': 125, 'coração': 126, 'preguiçoso': 127, 'foda': 128, 'todo': 129, 'liberto': 130, 'feiosa': 131, 'amar': 132, 'mulher': 133, 'salvos': 134, 'ontem': 135, 'pra': 136, 'judeu': 137, 'caso': 138, 'alvo': 139, 'antijudeu': 140, 'negras': 141, 'pro': 142, 'superiores': 143, 'complexo': 144, 'frios': 145, 'comigo': 146, 'vivo': 147, 'filha': 148, 'morrer': 149, 'fechar': 150, 'cabeça': 151, 'senzalado': 152, 'mundo': 153, 'forte': 154, 'porcos': 155, 'longa': 156, 'cabelo': 157, 'odiar': 158, 'narigudo': 159, 'brasil': 160, 'podre': 161, 'cair': 162, 'antinegro': 163, 'fogo': 164, 'melhores': 165, 'gato': 166, 'existir': 167, 'brancos': 168, 'nunca': 169, 'contrato': 170, 'querer': 171, 'judias': 172, 'louco': 173, 'dessa': 174, 'humano': 175, 'gentil': 176, 'bomba': 177, 'porco': 178, 'branco': 179, 'senzala': 180, 'assassino': 181, 'índia': 182, 'conversar': 183, 'importar': 184, 'namorar': 185, 'escravo': 186, 'caga': 187, 'ridículo': 188, 'sujeitar': 189, 'americano': 190, 'maluco': 191, 'neve': 192, 'corrupto': 193, 'saida': 194, 'gente': 195, 'amo': 196, 'pretos': 197, 'escravizar': 198, 'inferiores': 199, 'sincera': 200, 'fome': 201, 'detestar': 202, 'latinoamericanos': 203, 'jamais': 204, 'morra': 205, 'jumentos': 206, 'assassinos': 207, 'cafajeste': 208, 'triste': 209, 'correr': 210, 'morressem': 211, 'ser': 212, 'africano': 213, 'atividade': 214, 'crente': 215, 'perfeitas': 216, 'povo': 217, 'quer': 218, 'tão': 219, 'lamentável': 220, 'louca': 221, 'safar': 222, 'fofa': 223, 'dever': 224, 'linda': 225, 'viado': 226, 'crentes': 227, 'negros': 228, 'palhaço': 229, 'aí': 230, 'covarde': 231, 'abençoar': 232, 'saber': 233, 'volta': 234, 'bandido': 235, 'viver': 236, 'inútil': 237, 'ridícula': 238, 'parecer': 239, 'japonês': 240, 'filhos': 241, 'nascer': 242, 'parte': 243, 'pássaro': 244, 'jumentas': 245, 'branca': 246, 'negro': 247, 'preto': 248, 'filhas': 249, 'paraíso': 250, 'lembrar': 251, 'menina': 252, 'vida': 253, 'lindos': 254, 'sair': 255, 'áfrica': 256, 'lembre': 257, 'tempo': 258, 'besta': 259, 'vir': 260, 'inteligente': 261, 'calorosos': 262, 'lindas': 263, 'queimar': 264, 'ter': 265, 'esquerdistas': 266, 'chineses': 267, 'vacilão': 268, 'caminho': 269, 'nego': 270, 'legal': 271, 'todos': 272, 'pele': 273, 'africanos': 274, 'pegar': 275, 'verme': 276, 'sociedade': 277, 'devir': 278, 'pretas': 279, 'morte': 280, 'natureza': 281, 'judeus': 282, 'jogar': 283, 'presidente': 284, 'qualquer': 285, 'longe': 286, 'entrada': 287, 'russos': 288, 'inferior': 289, 'grande': 290, 'prevalecer': 291, 'fazer': 292, 'europeu': 293, 'tirar': 294, 'cagar': 295, 'pilantra': 296, 'real': 297, 'desgraça': 298, 'árabes': 299, 'neles': 300, 'todas': 301, 'arrombar': 302, 'feioso': 303, 'hoje': 304, 'pessoa': 305, 'disso': 306, 'sexy': 307, 'derrubar': 308, 'deixar': 309, 'pego': 310, 'ferrar': 311, 'casar': 312, 'puta': 313, 'escravidão': 314, 'demais': 315, 'aqui': 316, 'preferir': 317, 'menino': 318, 'desgraçar': 319, 'macaca': 320, 'burro': 321, 'bolsonaristas': 322, 'hitler': 323, 'cores': 324, 'liberdade': 325, 'loira': 326, 'coitar': 327, 'ideia': 328}
    
    #Criando objeto spacy
    nlp = spacy.load('pt_core_news_sm')
    
    #Convertendo tipo de dado
    discurso_spacy = nlp(DISCURSO)

    #Lemmatização
    discurso_spacy_lemmatizado = []
    for token in discurso_spacy:
        if token.pos_ in ['VERB', 'ADVERB']:
            discurso_spacy_lemmatizado.append(str(token).lower())
        else:
            discurso_spacy_lemmatizado.append(str(token).lower())
        
    #Retirando pontos
    sentenca_sem_pontuacao = []
    for token in discurso_spacy_lemmatizado:
        if str(token) not in string.punctuation:
            sentenca_sem_pontuacao.append(token)

    #Remoção de stopwords:
    sentenca_limpa = []
    stpw = nltk.corpus.stopwords.words('portuguese')
    for token in sentenca_sem_pontuacao:
        if str(token) not in stpw:
            sentenca_limpa.append(token)

    #Função para vetorização de discursos
    def vetorizacao2(texto):
        vetor = [0] * total_de_palavras
        for token in texto:
            token = str(token).lower()
            if token in palavra_posicao:
                posicao = palavra_posicao[token]
                vetor[posicao] += 1
        return vetor

         
    #Importação do modelo
    with open('identificador_multinomial_versao_3.pkl', 'rb') as file:
        modelo = pickle.load(file)
         
    #Programa para classificação do discurso
    vetor = vetorizacao2(sentenca_limpa)
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
