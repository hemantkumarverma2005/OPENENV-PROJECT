import json
import base64
from env.openenv_wrapper import SocialContractOpenEnv
from demo import smart_policy

task_id = "task4_stagflation"
env = SocialContractOpenEnv(task_id, seed=1337)
obs = env.reset()
done = False
while not done:
    action = smart_policy(obs)
    obs, reward, done, info = env.step(action)

state = env.state()
b64_img = state["visual_dashboard"]
with open("dashboard.png", "wb") as f:
    f.write(base64.b64decode(b64_img))
print("dashboard.png saved")
