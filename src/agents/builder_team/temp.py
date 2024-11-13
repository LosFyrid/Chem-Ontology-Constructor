import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 输出所有环境变量
for key, value in os.environ.items():
    print(f"{key}: {value}")
