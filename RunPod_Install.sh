cd /workspace

git clone --depth 1 https://github.com/comfyanonymous/ComfyUI

cd /workspace/ComfyUI

git reset --hard

git stash

git pull --force

python -m venv venv

source venv/bin/activate

python -m pip install --upgrade pip

pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

cd custom_nodes


git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager


# delete below part if you dont need ComfyUI_IPAdapter_plus - yes i recommend to delete this part
git clone --depth 1 https://github.com/cubiq/ComfyUI_IPAdapter_plus


# delete below part if you dont need ComfyUI-ReActor - yes i recommend to delete this part
git clone --depth 1 https://github.com/Gourieff/ComfyUI-ReActor


# delete below part if you dont need ComfyUI-GGUF - I don't recommend
git clone --depth 1 https://github.com/city96/ComfyUI-GGUF


# delete below part if you dont need ComfyUI-Impact-Pack - I don't recommend
git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Pack


# delete below part if you dont need sampler like res_2s and scheduler like beta57 - I don't recommend
git clone --depth 1 https://github.com/ClownsharkBatwing/RES4LYF


cd ComfyUI-Manager
git stash
git reset --hard
git pull --force
pip install -r requirements.txt
cd ..

# delete below part if you dont need ComfyUI_IPAdapter_plus - yes i recommend to delete this part
cd ComfyUI_IPAdapter_plus
git stash
git reset --hard
git pull --force
cd ..

# delete below part if you dont need ComfyUI-ReActor - yes i recommend to delete this part
cd ComfyUI-ReActor
git stash
git reset --hard
git pull --force
python install.py
pip install -r requirements.txt
cd ..

# delete below part if you dont need ComfyUI-GGUF - I don't recommend
cd ComfyUI-GGUF
git stash
git reset --hard
git pull --force
pip install -r requirements.txt
cd ..

# delete below part if you dont need ComfyUI-Impact-Pack - I don't recommend
cd ComfyUI-Impact-Pack
git stash
git reset --hard
git pull --force
python install.py
pip install -r requirements.txt
cd ..

# delete below part if you dont need sampler like res_2s and scheduler like beta57 - I don't recommend
cd RES4LYF
git stash
git reset --hard
git pull --force
pip install -r requirements.txt
cd ..






cd ..

echo Installing requirements...

pip install -r requirements.txt

pip uninstall xformers --yes

pip install https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/flash_attn-2.8.2-cp310-cp310-linux_x86_64.whl

pip install https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/xformers-0.0.33+c159edc0.d20250906-cp39-abi3-linux_x86_64.whl

pip install https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/sageattention-2.2.0.post4-cp39-abi3-linux_x86_64.whl

pip install https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/insightface-0.7.3-cp310-cp310-linux_x86_64.whl

cd ..

pip install -r requirements.txt

apt update

apt install psmisc

