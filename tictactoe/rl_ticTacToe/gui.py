import PySimpleGUI as sg
import threading
import multiprocessing as mp
import numpy as np
from .agent import Tictactoe
sg.theme('Black')
agent = Tictactoe()

layout = [
    [sg.Text('',size = (20,1) , key = 'end_state')],
    
   
    [sg.Button("",size = (5,1),enable_events = True,key = 'a'),sg.Button("",size = (5,1),enable_events = True,key = 'b'),sg.Button("",size = (5,1),enable_events = True,key = 'c')],
    [sg.Button("",size = (5,1),enable_events = True,key = 'd'),sg.Button("",size = (5,1),enable_events = True,key = 'e'),sg.Button("",size = (5,1),enable_events = True,key = 'f')],
    [sg.Button("",size = (5,1),enable_events = True,key = 'g'),sg.Button("",size = (5,1),enable_events = True,key = 'h'),sg.Button("",size = (5,1),enable_events = True,key = 'i')],
   
    
    [sg.Frame(layout=[
        [sg.Text('Training Intensity:')],
        [sg.Slider(default_value = 1000,range=(0, 10000000), size=(21,10), orientation='h', key='training_intensity')],

        [sg.Text('Alpha:',size = (len('Epsilon:'),1)),sg.Input(size = (5,1),key = 'alpha',default_text='0.99'),sg.Text('Gamma:'),sg.Input(size = (5,1),key = 'gamma',default_text='0.6')],
        [sg.Text('Epsilon:'),sg.Input(size = (5,1),key = 'epsilon',default_text='0.4')] ,
        [sg.Text('Max Processors:'),sg.Input(size = (5,1),key = 'cpus' ,default_text = str(mp.cpu_count()) )],
        [sg.Frame(layout =[
        [sg.Text('win:',size = (len('draw:'),1)),sg.Input(size = (5,1),key = 'win',default_text='100'),sg.Text('lose:'),sg.Input(size = (5,1),key = 'lose',default_text='-100')],
        [sg.Text('draw:',size = (len('draw:'),1)),sg.Input(size = (5,1),key = 'draw',default_text='-50'),sg.Text('play:'),sg.Input(size = (5,1),key = 'play',default_text='-1')],
        ],title = "rewards")],
        [sg.Text('',size= (20,1),key='training_status')],
    ],title = 'Parameters') ],
    
    [sg.Button('Train me'),sg.Button('exit'),sg.Button('reset')],
    
      
]

window = sg.Window('TicTacToe',layout,element_justification='center')
#              player empty agent
player_map = {1:'HUMAN' , 2:'AGENT SMITH'}
symbol_map = {'o':1 , '':0 , 'x':2  }



def evaluate(who,button_states):
    
    plays = np.where(np.array(button_states) == who)[0]
    if agent.check(plays,agent.wins):
        window['end_state'].update(f'{player_map[who]} WINS')
        return 0
    elif 0 not in button_states:
        window['end_state'].update('DRAW')
        return 0
    
    return 1
def run_game():
    isTraining = False
    while 1:

        event, value = window.read(timeout = 100)

        if event == 'exit':
            window.close()
            break
        elif event == 'reset':
            [window[chr(x)].update('',disabled = False) for x in range(ord('a'),ord('i')+1)]
            [window[chr(x)].update(button_color = ('black','white'))  for x in range(ord('a'),ord('i')+1)]
            window['end_state'].update('')   

        elif event == 'Train me':
            steps = int(value['training_intensity'])
            alpha = float(value['alpha'])
            gamma = float(value['gamma'])
            epsilon = float(value['epsilon'])
            win = int(value['win'])
            draw = int(value['draw'])
            lose = int(value['lose'])
            play = int(value['play'])
            cpus = int(value['cpus'])
            t1 = threading.Thread(target= agent.train , args = (steps,cpus,gamma,alpha,epsilon,win,draw,lose,play) , daemon = True)
            t1.start()
            window['training_status'].update('...Training....')
    #         agent.train(100000)
            isTraining = True
            window['Train me'].update(disabled = True)
        elif event != None and event.isalpha():
            window[event].update('o')
            window[event].update(disabled = True)
            window[event].update(button_color = ('black','red') )
            button_states = [symbol_map[window[chr(x)].GetText()]  for x in range(ord('a'),ord('i')+1)]
            if not evaluate(1,button_states):
                [window[chr(x)].update(button_color = ('black','white'),disabled=True)  for x in range(ord('a'),ord('i')+1)]
                continue

            agent_play = agent.agent_play(button_states)
            window[agent_play].update('x')
            window[agent_play].update(disabled = True)
            window[agent_play].update(button_color = ('black','blue'))
            button_states = [symbol_map[window[chr(x)].GetText()]  for x in range(ord('a'),ord('i')+1)]
            if not evaluate(2,button_states):
                [window[chr(x)].update(button_color = ('black','white'),disabled=True)  for x in range(ord('a'),ord('i')+1)]
                continue

        if isTraining:

            try:
                agent.training_done.get_nowait()
                window['training_status'].update('...DONE....')
                isTraining = False
                window['Train me'].update(disabled = False)
            except:
                pass
        
    
