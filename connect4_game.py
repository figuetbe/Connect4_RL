# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 22:07:06 2019

@author: figu
"""

import csv
import sys
from agents import agents as ag

def main():
   menu()

def menu():
    print("************Welcome to Connect4-RL**************")
    choice = input("""
                      1: Play Human vs Human
                      2: Play vs AI
                      3: Train AI
                      4: Rage Quit

                      Please enter your choice: """)

    if choice == "1":
        ag.play_human()
    elif choice == "2":
        ag.play_ai()
    elif choice == "3":
        ag.train_td()
    elif choice=="4":
        sys.exit
    else:
        print("You must select a valid option")
        print("Please try again")
        menu()
    
if __name__ == '__main__':
    main()