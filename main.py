import json
import pyro
from pyro.distributions import Bernoulli,constraints,Beta,Binomial
from pyro.infer import SVI, Trace_ELBO,Importance, EmpiricalMarginal
from pyro.optim import Adam
import task_managing_queue
import torch

class DATA:
    def __init__(self):
        self.loadAllData()
        task = task_managing_queue.task_queue()
        self.love = round(self.take_loveforit()/100,4) 
        if self.love<0.5:
            accuracy = 5
        else:
            accuracy = 15 
        task.start_work()
        if self.love!=0:
            love_result = task.add_task(self.do_svi_for_love,(self.love,accuracy,2000))
  
        self.prev_success = self.take_previoussuccess()#total , success
        prev_success_result = task.add_task( self.do_svi_for_prev_success)
        
        self.country = self.take_country()
        self.gdp = [0,0]
        self.gdp[0] = self.gdp_list[self.country]
        self.gdp[1] = self.gdp_list["MAX"]
        self.popularity = self.take_popularity()
        self.job = self.have_job()
        self.loan = self.have_loan()
        
        task.stop_work()
        print("Wait compiling inputs")
        if self.love!=0:
            self.love = love_result.get()
            
        self.prev_success = prev_success_result.get()
        task.shutdown()
        
        
    
    def loadAllData(self):
        with open("country.json") as f:
            self.gdp_list = json.load(f)

    def take_country(self):
        print("Enter your country: ")
        while True:
            country = input("")
            country = country.lower().replace(" ","")
            if country not in self.gdp_list:
                print("Country do not exist: ")
                continue
            return country

    def have_loan(self):
        print("Did you take any loan: ")
        while True:
            yesORno = input("yes or no:  ").lower().replace(" ","")
            if yesORno not in ["yes","no"]:
                print("reply with only ",end="")
                continue    
            if yesORno=="yes":
                return 1
            else:
                return 0
    
    def have_job(self):
        job = 0
        print("DO you have a job: ")
        while True:
            yesORno = input("yes or no ?").lower().replace(" ","")
            if yesORno not in ["yes","no"]:
                print("reply with only ",end="")
                continue    
            if yesORno=="yes":
                print("What is your income in dollors(monthly): ")
                while True:
                    try:
                        choice = float(input())
                        if choice<0:
                            print("number must be >= 0")    
                        else:
                            job = choice    
                            break
                    except ValueError:
                        print("only enter numbers")
                    except Exception as e:
                        print(e)
                        print("Something went wrong") 
                monthly_gdp = self.gdp[0] / 12  # Convert total GDP to monthly


                if job <= 0.005 * monthly_gdp:        # Poor: ~0.5% of monthly GDP
                    job = 1
                elif job <= 0.015 * monthly_gdp:      # Lower-Middle: 1.5%
                    job = 2
                elif job <= 0.03 * monthly_gdp:       # Middle-Middle: 3%
                    job = 3
                elif job <= 0.06 * monthly_gdp:       # Upper-Middle: 6%
                    job = 4
                else:                                    # Rich: more than 6% of monthly GDP
                    job = 5         
                         

                break        
                
            else:
                yesORno = 0
                job = yesORno 
                break
                
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
            print("Enter your choice (1,2,3,4,5): ")
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
        print(f"How much {"%"} do you love it: ")
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
        print("How many business have you started before: ")
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
                elif choice>answer[0]:
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
        pyro.clear_param_store()
        #model part
        def make_model(observed, accuracy):
            def modelEXE():
                true_prob = pyro.sample("P_love", Beta(2.0, 2.0))
                
                # We assume observed proportion is a noisy measurement of the true probability
                # So we model it as Beta-distributed around true_prob (with fixed concentration)
                pyro.sample("observed_P_love", Beta(true_prob * accuracy, (1 - true_prob) * accuracy),  # 20 = confidence in proportion
                                    obs=torch.tensor(float(observed)))  # the observed proportion (e.g., 79%)
            return modelEXE
        #guide part
        def guideEXE():
            alpha_q = pyro.param("alpha_P_love", torch.tensor(2.0), constraint=constraints.positive)
            beta_q = pyro.param("beta_P_love", torch.tensor(2.0), constraint=constraints.positive)
            pyro.sample("P_love", Beta(alpha_q, beta_q))
            
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


    def do_svi_for_prev_success(self,steps=3000):
        pyro.clear_param_store()
        #model part
        def make_model():
            def modelEXE():
                possibleprob = pyro.sample("P_prev_success",Beta(2.0,2.0))#prob of not
                pyro.sample("observed",Binomial(total_count=self.prev_success[0],probs = possibleprob),obs=torch.tensor(float(self.prev_success[1])))
                
            return modelEXE
        #guide part
        def guideEXE():
            alpha_q = pyro.param("alpha_P_prev_success", torch.tensor(2.0), constraint=constraints.positive)
            beta_q = pyro.param("beta_P_prev_success", torch.tensor(2.0), constraint=constraints.positive)
            pyro.sample("P_prev_success", Beta(alpha_q, beta_q))
            
        #svi part
        modelEXE = make_model()
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
    def __init__(self,job,loan,choice,love,gdp,prev_success):
        self.have_job = job #0
        self.loan = loan #0 or 1
        self.choice = choice #1,2,3,4,5 #popularity
        self.love = love
        self.gdp = gdp # countryGDP max
        self.prev_success = prev_success

    
    
        
    def network(self):
        P_job_table = [0.3,0.45,0.56,0.66,0.8,0.91]#No job  poor mediumpoor middlefair middleupper upper
        
        P_job = P_job_table[self.have_job]   
        job = pyro.sample("job", Bernoulli(P_job), obs=torch.tensor(float(self.have_job))) 
        
        P_loan = torch.where(job==torch.tensor(1.0),0.4,0.6)
        loan = pyro.sample("loan",Bernoulli(P_loan),obs=torch.tensor(float(self.loan)))#will he repay it
        i,j = int(job.item()),int(loan.item())
        cpt_job_loan = [[0.2,0.4],
                        [0.8,0.6]]
        P_investment = cpt_job_loan[i][j]
        input_investment = pyro.sample("investment",Bernoulli(P_investment))
        
        
        
        P_love = self.love
        love = pyro.sample("love",Bernoulli(P_love))
        
        P_prev_success = self.prev_success
        prev_success = pyro.sample("prev_success",Bernoulli(P_prev_success))
        
        i,j = int(love.item()),int(prev_success.item())
        cpt_hardwork = [[0.40,0.60],[0.54,0.67]]
        P_hardwork = cpt_hardwork[i][j]
        hardwork = pyro.sample("hardwork",Bernoulli(P_hardwork))        
        
        
        
        P_popularity = [None,0.3,0.4,0.7,0.4,0.4][self.choice] #none is just a place holder
        popularity = pyro.sample("popularity",Bernoulli(P_popularity))
        P_gdp = (self.gdp[0]/self.gdp[1]) # gdp/max  
        gdp = pyro.sample("gdp",Bernoulli(P_gdp))
        
        gdp_value,j = int(popularity.item()),int(gdp.item())
        cpt_market = [[0.23,0.5],[0.47,0.71]]
        P_market = cpt_market[j][gdp_value]
        market = pyro.sample("market",Bernoulli(P_market))
        
        i,j,k = int(market.item()),int(hardwork.item()),int(input_investment.item())
        cpt_success = [
                        [  # gdp = 0
                            [  # market = 0
                                [0.16, 0.39],  # hardwork = 0 → investment = 0, 1
                                [0.29, 0.48],  # hardwork = 1 → investment = 0, 1
                            ],
                            [  # market = 1
                                [0.34, 0.60],# hardwork = 0 → investment = 0, 1
                                [0.50, 0.77],# hardwork = 1 → investment = 0, 1
                            ],
                        ],
                        [  # gdp = 1
                            [  # market = 0
                                [0.40, 0.50],# hardwork = 0 → investment = 0, 1
                                [0.45, 0.86],# hardwork = 1 → investment = 0, 1
                            ],
                            [  # market = 1
                                [0.63, 0.81],# hardwork = 0 → investment = 0, 1
                                [0.75, 0.94],# hardwork = 1 → investment = 0, 1
                            ],
                        ],
                    ]
        P_success = cpt_success[gdp_value][i][j][k]
        success = pyro.sample("success",Bernoulli(P_success))
        
        return {
            "job": job.item(),
            "loan": loan.item(),
            "investment": input_investment.item(),
            "love": love.item(),
            "prev_success": prev_success.item(),
            "hardwork": hardwork.item(),
            "popularity": popularity.item(),
            "gdp": gdp.item(),
            "market": market.item(),
            "success": success.item()
        }
        
    def infer_success(self, num_samples=1000):
        # Assume self.network already includes observations for conditioning
        posterior = Importance(self.network, num_samples=num_samples).run()
        
        # Get all "success" samples (bools or 0/1 floats)
        success_vals = [
            posterior.exec_traces[i].nodes["success"]["value"]
            for i in range(num_samples)
        ]
        
        # Convert to floats and average
        success_prob = sum(float(val) for val in success_vals) / num_samples
        return round(success_prob, 4)
        
if __name__ == "__main__":
        dataclass = DATA()
        job = dataclass.job
        loan = dataclass.loan
        popularity = dataclass.popularity
        love = dataclass.love
        gdp = dataclass.gdp
        prev_success = dataclass.prev_success
        
        Bayesian_network = mainpart(job,loan,popularity,love,gdp,prev_success) 
                   
        answer = Bayesian_network.infer_success(1500)
        print(f"Your business have a {answer}{"%"} chance of success")
        
        
            