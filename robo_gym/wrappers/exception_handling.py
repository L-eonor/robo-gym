import gym
from grpc import RpcError
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
import json
import numpy as np
import time

path_to_file="/home/leonor/logs/pick_and_place/reset_logs.txt"
path_to_step_log="/home/leonor/logs/pick_and_place/step"
current_path_to_step_log=None


class ExceptionHandling(gym.Wrapper):

    def step(self, action):
        try:
            observation, reward, done, info = self.env.step(action)

            f = open(current_path_to_step_log, "a")
            f.write("Observation:\n")
            np.savetxt(f, observation)
            f.write("Action:\n")
            np.savetxt(f, action)
            f.write("Reward:"+str(reward)+'\n')
            f.write("Done:"+str(done)+'\n')
            f.write("info:"+info['final_status']+'\nsucess:'+str(info['is_success'])+'\n')
            f.close()

            if done:
                f = open(path_to_file, "a")
                f.write("Observation:\n")
                np.savetxt(f, observation)
                f.write("Reward:"+str(reward)+'\n')
                f.write("Done:"+str(done)+'\n')
                f.write("info:"+info['final_status']+'\nsucess:'+str(info['is_success'])+'\n')
                
                f.close()
            return observation, reward, done, info
        except (RpcError, InvalidStateError, RobotServerError) as e:
            if InvalidStateError and not (RobotServerError):
                print('Invalid state error. Restarting Robot Server and reseting world...')

                observation=self.env.reset()
                reward=-10.0
                done=True
                info={"Exception":True, "ExceptionType": e}



                f = open(path_to_file, "a")
                f.write("Observation:\n")
                np.savetxt(f, observation)
                f.write("Reward:"+str(reward)+'\n')
                f.write("Done:"+str(done)+'\n')
                f.write("info:"+ 'Exception; '+str(e)+'\n')
                f.close()
            
                
       
                f = open(current_path_to_step_log, "a")
                f.write("Observation:\n")
                np.savetxt(f, observation)
                f.write("Action:\n")
                np.savetxt(f, action)
                f.write("Reward:"+str(reward)+'\n')
                f.write("Done:"+str(done)+'\n')
                f.write("info:"+str(e)+'\n')
                f.close()


                return observation, reward, done, info
            else:
                print('Error occurred while calling the step function. Restarting Robot server ...')
                self.env.restart_sim()
                
                #observation=self.env.observation_space.sample()
                observation=self.env.reset()
                reward=-10.0
                done=True
                info={"Exception":True, "ExceptionType": e}


                f = open(path_to_file, "a")
                f.write("Observation:\n")
                np.savetxt(f, observation)
                f.write("Reward:"+str(reward)+'\n')
                f.write("Done:"+str(done)+'\n')
                f.write("info:"+ 'Exception; '+str(e)+'\n')
                f.close()
            
    
                f = open(current_path_to_step_log, "a")
                f.write("Observation:\n")
                np.savetxt(f, observation)
                f.write("Action:\n")
                np.savetxt(f, action)
                f.write("Reward:"+str(reward)+'\n')
                f.write("Done:"+str(done)+'\n')
                f.write("info:"+str(e)+'\n')
                f.close()

                return observation, reward, done, info

    def reset(self, **kwargs):
        global current_path_to_step_log
        current_path_to_step_log=path_to_step_log+str(round(time.time() * 1000))+'.txt'
        
        for i in range(5):
            try:
                return self.env.reset(**kwargs)
            except (RpcError, InvalidStateError, RobotServerError):
                print('Error occurred while calling the reset function. Restarting Robot server ...')
                self.env.restart_sim()
        raise Exception("Failed 5 tentatives to reset environment.")
    