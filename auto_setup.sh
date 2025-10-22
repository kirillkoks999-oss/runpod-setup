#!/bin/bash

set -e

# Устанавливаем токены и переменные окружения
export HF_HOME="/workspace"
export HUGGING_FACE_HUB_TOKEN="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"

echo "=========================================="
echo "ШАГ 1: Клонирование конфигурации"
echo "=========================================="
cd /workspace

if [ -d "runpod-setup" ]; then
    echo "Репо уже существует, обновляем..."
    cd runpod-setup
    git pull
    cd /workspace
else
    git clone https://github.com/kirillkoks999-oss/runpod-setup.git
fi

cp -r runpod-setup/* /workspace/
chmod +x /workspace/RunPod_Install.sh

echo "=========================================="
echo "ШАГ 2: Запуск RunPod_Install.sh"
echo "=========================================="
cd /workspace
nohup ./RunPod_Install.sh > /workspace/install.log 2>&1 &
INSTALL_PID=$!

echo "Установка запущена в фоне (PID: $INSTALL_PID). Ждем завершения..."
while kill -0 $INSTALL_PID 2>/dev/null; do
    echo "Установка продолжается..."
    sleep 10
done

echo "✅ RunPod_Install.sh завершен"

echo "=========================================="
echo "ШАГ 3: Установка системных зависимостей"
echo "=========================================="

# FFmpeg
echo "Устанавливаем FFmpeg..."
cd /workspace
rm -rf ffmpeg-N-118385-g0225fe857d-linux64-gpl.tar.xz ffmpeg-N-118385-g0225fe857d-linux64-gpl
wget -q https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2025-01-31-12-58/ffmpeg-N-118385-g0225fe857d-linux64-gpl.tar.xz
tar xf ffmpeg-N-118385-g0225fe857d-linux64-gpl.tar.xz --no-same-owner
mv ffmpeg-N-118385-g0225fe857d-linux64-gpl/bin/ffmpeg /usr/local/bin/
mv ffmpeg-N-118385-g0225fe857d-linux64-gpl/bin/ffprobe /usr/local/bin/
chmod +x /usr/local/bin/ffmpeg
chmod +x /usr/local/bin/ffprobe
echo "✅ FFmpeg установлен"

# Cloudflared
echo "Устанавливаем Cloudflared..."
rm -rf cloudflared-linux-amd64.deb
wget -q https://github.com/cloudflare/cloudflared/releases/download/2025.7.0/cloudflared-linux-amd64.deb
dpkg -i cloudflared-linux-amd64.deb
echo "✅ Cloudflared установлен"

echo "=========================================="
echo "ШАГ 4: Установка SwarmUI"
echo "=========================================="

cd /workspace
if [ ! -d "SwarmUI" ]; then
    echo "Клонируем SwarmUI..."
    git clone --depth 1 https://github.com/mcmonkeyprojects/SwarmUI
    git clone --depth 1 https://github.com/Fannovel16/ComfyUI-Frame-Interpolation SwarmUI/src/BuiltinExtensions/ComfyUIBackend/DLNodes/ComfyUI-Frame-Interpolation
    git clone --depth 1 https://github.com/welltop-cn/ComfyUI-TeaCache SwarmUI/src/BuiltinExtensions/ComfyUIBackend/DLNodes/ComfyUI-TeaCache
else
    echo "SwarmUI существует, обновляем..."
fi

cd SwarmUI
git reset --hard
git stash
git pull

# Установка .NET
echo "Устанавливаем .NET..."
cd launchtools
if [ ! -f "dotnet-install.sh" ]; then
    wget -q https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh
    chmod +x dotnet-install.sh
fi
./dotnet-install.sh --channel 8.0 --runtime aspnetcore > /dev/null 2>&1
./dotnet-install.sh --channel 8.0 > /dev/null 2>&1
echo "✅ SwarmUI установлен"

echo "=========================================="
echo "ШАГ 5: Установка Python библиотек"
echo "=========================================="
cd /workspace
pip install -q gradio huggingface_hub[cli] hf_transfer hf_xet
echo "✅ Python библиотеки установлены"

echo "=========================================="
echo "ШАГ 6: Скачивание моделей"
echo "=========================================="

# Создаем папки для моделей
mkdir -p /workspace/ComfyUI/models/checkpoints
mkdir -p /workspace/ComfyUI/models/loras

# Скачивание checkpoint через huggingface-cli
echo "Скачиваем Lily.safetensors через HuggingFace CLI..."
cd /workspace/ComfyUI/models/checkpoints/
if [ ! -f "Lily.safetensors" ]; then
    huggingface-cli download rillky/Lily Lily.safetensors --local-dir . --token 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
    echo "✅ Lily.safetensors скачан"
else
    echo "Lily.safetensors уже существует"
fi

# Скачивание LoRA с Civitai
echo "Скачиваем UltraRealistic_Flux LoRA..."
cd /workspace/ComfyUI/models/loras/
if [ ! -f "UltraRealistic_Flux.safetensors" ]; then
    wget "https://civitai.com/api/download/models/1026423?type=Model&format=SafeTensor" -O UltraRealistic_Flux.safetensors
    echo "✅ UltraRealistic_Flux.safetensors скачана"
else
    echo "UltraRealistic_Flux.safetensors уже существует"
fi

echo "=========================================="
echo "ШАГ 7: Запуск SwarmUI"
echo "=========================================="

cd /workspace/SwarmUI
echo "Запускаем SwarmUI в фоне..."
nohup ./launch-linux.sh --launch_mode none --cloudflared-path cloudflared --port 7861 > /workspace/swarm.log 2>&1 &
echo "✅ SwarmUI запущен в фоне"

echo ""
echo "Ждем 15 секунд для получения ссылки SwarmUI..."
sleep 15

# Извлекаем ссылку на SwarmUI из логов
echo ""
echo "========================================"
echo "🔗 ССЫЛКА НА SWARMUI:"
echo "========================================"
grep -i "cloudflare\|https://.*trycloudflare.com" /workspace/swarm.log | tail -5
echo "========================================"
echo ""
echo "Полные логи SwarmUI: /workspace/swarm.log"
echo "Команда для просмотра: tail -f /workspace/swarm.log"
echo ""

echo "=========================================="
echo "ШАГ 8: Запуск Gradio приложения"
echo "=========================================="

cd /workspace

echo ""
echo "========================================"
echo "🚀 Запускаем Gradio приложение..."
echo "========================================"
echo "Ссылка появится ниже:"
echo ""

python -W ignore Downloader_Gradio_App.py --share

echo "=========================================="
echo "✅ ВСЕ ГОТОВО!"
echo "=========================================="
