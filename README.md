# Clear instructions to reproduce your results:

## 1. Clone the repository:
git clone https://github.com/TsaoPC/EAI_FinalProject_Group10.git   
cd EAI_FinalProject_Group10/  

## 2. Set up the environment:
sudo apt update  
sudo apt install python3-pip -y  
pip install --upgrade pip setuptools wheel
pip install meson==1.3.1  
sudo apt install -y pkg-config libcairo2-dev cmake build-essential
pip install -r requirements.txt  
 
## 3. Run the code:(confirm VLLM_AVAILABLE = True in line 13 for calculate tupt, VLLM_AVAILABLE = Flase for ppl. )   
python3 result.py  
