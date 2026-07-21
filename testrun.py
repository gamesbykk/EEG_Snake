from test2ai import Muse2LSTMProcessor
from test import SnakeGame, EEGSnakeRecorder
import pygame
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
pygame.init()
clock=pygame.time.Clock()
screen=pygame.display.set_mode((800,600))
snake=SnakeGame()
eeg=EEGSnakeRecorder()
eeg.initialize_stream()
running=1
lstm=Muse2LSTMProcessor().load_model()
le=LabelEncoder()
df=pd.read_csv('all_eeg_snake_data.csv')
le.fit(df['action'])
print(le.classes_)
df['action']=df['action'].replace("no_action", pd.NA).ffill()
le.fit(df['action'])
print(le.classes_)
le.fit(df['action'])
print(le.classes_)
ss=StandardScaler()
ss.fit_transform(df[['TP9','TP10','AF7','AF8']])
print(le.classes_)
data64=[]
while running:
    for event in pygame.event.get():
        if event.type==pygame.KEYDOWN:
            if event.key==pygame.K_r and snake.game_over:
                snake.reset_game()
    data64.extend(ss.transform(eeg._collect_sample()))
    if len(data64) > 256:
        while len(data64)!=256:
            data64.pop(0)
        input_data=np.array(data64).reshape(1, 256, 4)
        preds=lstm.predict(input_data)
        print(preds)
        dir=[(0,1), (-1,0), (1,0), (0,1)][np.argmax(preds[0])]
        if dir:
            snake.update_direction(dir)
    snake.update_game()
    snake.draw(screen)
    pygame.display.flip()
    clock.tick(10)
    