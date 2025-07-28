import json
import pyro
from pyro.distributions import Bernoulli,constraints,Beta
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import torch

#(i / x) * 100 >gdp
class DATA:
    def __init__(self):
        self.loadAllData()
        self.love = round(self.take_loveforit()/100,4)
        self.country = self.take_country()
        self.popularity = self.take_popularity()
        if self.love<50:
            accuracy = 5
        else:
            accuracy = 15    
        self.love = self.do_svi_for_love(self.love,accuracy,2000)
        self.prev_success = self.do_svi_for_prev_success()
        self.job = self.have_job()
        
        
        
        temp = self.take_previoussuccess()#total , success
        self.prev_success = (temp[1]/temp[0])
        
    
    def loadAllData(self):
        with open("country.json") as f:
            self.gdp = json.load(f)

    def take_country(self):
        print("Enter your country")
        while True:
            country = input("")
            country = country.lower().replace(" ","")
            if country not in self.gdp:
                print("Country do not exist")
                continue
            return country

    def have_job(self):
        job = [0,0]
        print("DO you have a job")
        while True:
            yesORno = input("yes or no ?").lower().replace(" ","")
            if yesORno not in ["yes","no"]:
                print("reply with only ",end="")
                continue    
            if yesORno=="yes":
                yesORno = 1
                print("What is your income in dollors(monthly)")
                while True:
                    try:
                        choice = float(input())
                        if choice<0:
                            print("number must be >= 0")    
                        else:
                            job[1] = choice    
                            break
                    except ValueError:
                        print("only enter numbers")
                    except Exception as e:
                        print(e)
                        print("Something went wrong") 
                if job[1]<=85:
                    job[1] = 1
                elif job<=200:
                    job[1] = 2
                elif job<=500:
                    job[1] = 3
                elif job<=1200:
                    job[1] = 4
                else:
                    job[1] = 5            
                         

                break        
                
            else:
                yesORno = 0 
                break
                
        job[0] = yesORno
        return job
    
    def take_popularity(self):
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

    def take_loveforit(self):
        print(f"How much {"%"} do you love it ")
        while True:
            try:
                choice = float(input())
                if choice<0:
                    print("number must be >= 0")
                elif choice>100:
                    print("number must be <= 100")    
                else:
                    return choice    
            except ValueError:
                print("only enter numbers")
            except Exception as e:
                print(e)
                print("Something went wrong")    

    def take_previoussuccess(self):
        print("How many business have you started before:")
        answer = [0,0]
        while True:
            try:
                choice = int(input())
                if choice<0:
                    print("number must be >= 0")   
                else:
                    answer[0] = choice    
                    break
            except ValueError:
                print("only enter numbers")
            except Exception as e:
                print(e)
                print("Something went wrong")
        print("how many of them succeeded")        
        while True:
            try:
                choice = int(input())
                if choice<0:
                    print("number must be >= 0")   
                elif choice<answer[0]:
                    print(f"must be less than total number of business ({answer[0]})")
                else:
                    answer[1] = choice    
                    break
            except ValueError:
                print("only enter numbers")
            except Exception as e:
                print(e)
                print("Something went wrong")         
        return answer

    def do_svi_for_love(self,observed,accuracy,steps=3000):
        
        #model part
        def make_model(observed, accuracy):
            def modelEXE():
                true_prob = pyro.sample("P_love", Beta(2.0, 2.0))
                
                # We assume observed proportion is a noisy measurement of the true probability
                # So we model it as Beta-distributed around true_prob (with fixed concentration)
                pyro.sample("observed_P_love", Beta(true_prob * accuracy, (1 - true_prob) * accuracy),  # 20 = confidence in proportion
                                    obs=torch.tensor(observed))  # the observed proportion (e.g., 79%)
            return modelEXE
        #guide part
        def guideEXE():
            alpha_q = pyro.param("alpha_P_love", torch.tensor(2.0), constraint=constraints.positive)
            beta_q = pyro.param("beta_P_love", torch.tensor(2.0), constraint=constraints.positive)
            pyro.sample("P_love", Beta(alpha_q, beta_q))
            pyro.clear_param_store()
        #svi part
        modelEXE = make_model(observed, accuracy)
        svi = SVI(model=modelEXE, guide=guideEXE,
                optim=Adam({"lr": 0.01}), loss=Trace_ELBO())

        for _ in range(steps):
            svi.step()

        alpha_val = pyro.param("alpha_P_love").item()
        beta_val = pyro.param("beta_P_love").item()
        posterior = Beta(alpha_val, beta_val)
        sample_p = posterior.sample()
        return sample_p


    def do_svi_for_prev_success(self,observed,accuracy,steps=3000):
        
        #model part
        def make_model(observed, accuracy):
            def modelEXE():
                true_prob = pyro.sample("P_prev_success", Beta(2.0, 2.0))
                
                # We assume observed proportion is a noisy measurement of the true probability
                # So we model it as Beta-distributed around true_prob (with fixed concentration)
                pyro.sample("observed_P_prev_success", Beta(true_prob * accuracy, (1 - true_prob) * accuracy),  # 20 = confidence in proportion
                                    obs=torch.tensor(observed))  # the observed proportion (e.g., 79%)
            return modelEXE
        #guide part
        def guideEXE():
            alpha_q = pyro.param("alpha_P_prev_success", torch.tensor(2.0), constraint=constraints.positive)
            beta_q = pyro.param("beta_P_prev_success", torch.tensor(2.0), constraint=constraints.positive)
            pyro.sample("P_prev_success", Beta(alpha_q, beta_q))
            pyro.clear_param_store()
        #svi part
        modelEXE = make_model(observed, accuracy)
        svi = SVI(model=modelEXE, guide=guideEXE,
                optim=Adam({"lr": 0.01}), loss=Trace_ELBO())

        for _ in range(steps):
            svi.step()

        alpha_val = pyro.param("alpha_P_prev_success").item()
        beta_val = pyro.param("beta_P_prev_success").item()
        posterior = Beta(alpha_val, beta_val)
        sample_p = posterior.sample()
        return sample_p

    
class mainpart:
    def __init__(self,job,loan,choice,love):
        self.have_job = job #[have?,income]
        self.loan = loan #0 or 1
        self.choice = choice #1,2,3,4,5 #popularity
        self.love = love

    
    
        
    def network(self):
        P_job = [0.3,0.45,0.56,0.66,0.8,0.91]#No job  poor mediumpoor middlefair middleupper upper
        job = pyro.sample("job", Bernoulli(P_job), obs=torch.tensor(self.have_job)) 
        P_loan = torch.where(job==torch.tensor(1.0),0.4,0.6)
        loan = pyro.sample("loan",Bernoulli(P_loan),obs=torch.tensor(self.loan))#will he repay it
        i,j = int(job.item()),int(loan.item())
        cpt_job_loan = [[0.2,0.4],
                        [0.8,0.6]]
        P_investment = cpt_job_loan[i][j]
        input_investment = pyro.sample("investment",Bernoulli(P_investment))
        
        
        P_popularity = [None,0.3,0.4,0.7,0.4,0.4][self.choice]
        popularity = pyro.sample("popularity",Bernoulli(P_popularity))
        
        P_love = self.love
        love = pyro.sample("love",Bernoulli(P_love))
        
        P_prev_success = None
        prev_success = pyro.sample("prev_success",P_prev_success)
        cpt_hardwork = [[]]
        
        
        
        
        
        
                
    