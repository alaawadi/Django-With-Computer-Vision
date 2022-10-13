# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 23:47:01 2021

@author: E Alaa Wadi
"""

from random import randint
from timeit import default_timer


def generator():
    start = default_timer()

    total = 0
    index = 0
    index_0 = 0
    index_1 = 0

    choice = input('Do you want wish credit card "visa" or "mastercard"? ')

    if choice == 'visa' or choice == 'mastercard':
        if choice == 'visa':
            first_num = '4'

        if choice == 'mastercard':
            first_num = '5'
        print('OK')

    else:
        print('Only generate "visa" and "mastercard" credit card numbers.')

    amount = int(input('Type how many card numbers you want generate: '))

    for i in range(amount):
        left_num = str(randint(11111111111111, 99999999999999))
        first_num += left_num

        # LUHN ALGORITHM

        if first_num[0] == '4':

            for index in range(15):
                if index % 2 == 0:
                    total += int(first_num[index]) * 2
                else:
                    total += int(first_num[index])

            digit = 10 - (total % 10)
            if digit > 9:
                digit = 0

            card_num = first_num + str(digit)

        # MASTERCARD ALGORITHM

        if first_num[0] == '5':

            for index in range(15):
                if index % 2 == 0:
                    i = int(first_num[index]) * 2
                else:
                    i = int(first_num[index])
                if i > 9:
                    x = str(i)
                    index_0 += int(x[0])
                    index_1 += int(x[1])
                    y = index_0 + index_1
                else:
                    y = i
                total += y
                x = ''
                y = 0
                i = 0
                index_0 = 0
                index_1 = 0

            digit = 10 - (total % 10)
            if digit > 9:
                digit = 0

            card_num = first_num + str(digit)

        print(card_num)
        first_num = card_num[0]
        left_num = ''
        total = 0
        digit = 0
        card_num = ''

    stop = default_timer()
    print(f'Tempo de exec:{stop-start}')


if __name__ == "__main__":
    generator()