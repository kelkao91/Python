#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class Bank_Account:
    def __init__(self,initial=0):
        self.balance = initial
    
    def deposit(self,amount):
        self.balance += amount
        print("Amount Deposited: ",amount)
        
    def withdraw(self,amount):
        if amount <= self.balance:
            self.balance -= amount
            print('Withdrew: ',amount)
        else:
            print("Insufficient Balance. ")
            self.inquire()
    
    def inquire(self):
        print('Available Balance: ',self.balance)
        
    def __add__(self, other):
        total = self.balance + other.balance
        self.balance = 0
        other.balance = 0
        return Bank_Account(total)


# In[ ]:


acc1 = Bank_Account()
acc1.deposit(1000)

