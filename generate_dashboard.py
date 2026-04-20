import json
import base64
from env.openenv_wrapper import SocialContractOpenEnv
from demo import smart_policy

task_id = "task4_stagflation"
env = SocialContractOpenEnv(task_id, seed=1337)
obs = env.reset()
while not env.is_done:
    action = smart_policy(obs)
    obs = env.step(action)

state = env._full_state()
b64_img = state["visual_dashboard"]
output_paths = ["dashboard.png", "dashboard.generated.png"]
image_bytes = base64.b64decode(b64_img)
for path in output_paths:
    try:
        with open(path, "wb") as f:
            f.write(image_bytes)
        print(f"{path} saved")
        break
    except PermissionError:
        continue
else:
    raise PermissionError(
        "Could not write dashboard image. Close any app using dashboard.png "
        "or delete the locked file and rerun."
    )
