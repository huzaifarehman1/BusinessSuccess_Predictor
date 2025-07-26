import json
import pyro
from pyro.distributions import Bernoulli
import torch
#(i / x) * 100 >gdp
class DATA:
    def loadAllData(self):
        with open("country.json") as f:
            self.gdp = json.load(f)

def take_country(data:DATA):
    print("Enter your country")
    while True:
        country = input("")
        country = country.lower().replace(" ","")
        if country not in data.gdp:
            print("Country do not exist")
            continue
        return country

def have_job():
    print("DO you have a job")
    while True:
        yesORno = input("yes or no ?").lower().replace(" ","")
        if yesORno not in ["yes","no"]:
            print("reply with only ",end="")
            continue    
        if yesORno=="yes":
            yesORno = 1
            #do this
            
            
        else:
            yesORno = 0     

def take_popularity():
    print("what is the popularity of the business your doing as a whole")
    print("OPTIONS:")
    print("1:low popularity")
    print("2:medium low popularity")
    print("3:medium fair popularity")
    print("4:medium high popularity")
    print("5:high popularity")
    
    while True:
        print("Enter your choice (1,2,3,4,5)")
        try:
            choice = int(input())
            if choice not in [1,2,3,4,5]:
                print("invalide choice")
                continue
            return choice
        except ValueError:
            print("only integer allowed")
            continue
        except Exception as e:
            print(e)
            print("something went wrong:(") 
            continue   

def take_loveforit():
    pass

def take_previoussuccess():
    pass


class mainpart:
    def __init__(self,job,loan,choice):
        self.have_job = job #0 or 1
        self.loan = loan #0 or 1
        self.choice = choice #1,2,3,4,5
    def network(self):
        job = pyro.sample("job", Bernoulli(0.6), obs=torch.tensor(self.have_job)) 
        P_loan = torch.where(job==torch.tensor(1.0),0.4,0.6)
        loan = pyro.sample("loan",Bernoulli(P_loan),obs=torch.tensor(self.loan))#will he repay it
        i,j = int(job.item()),int(loan.item())
        cpt_job_loan = [[0.2,0.4],
                        [0.8,0.6]]
        P_investment = cpt_job_loan[i][j]
        input_investment = pyro.sample("investment",Bernoulli(P_investment))
        
        
        P_popularity = [None,0.3,0.4,0.7,0.4,0.4][self.choice]
        popularity = pyro.sample("popularity",Bernoulli(P_popularity))
        
        
        
        
        
        
                
    