#!/bin/bash

# ============================================
# АВТОМАТИЧЕСКАЯ УСТАНОВКА RUNPOD
# ============================================

set -e  # Остановка при ошибке

echo "=========================================="
echo "ШАГ 1: Клонирование конфигурации"
echo "=========================================="
export HF_HOME="/workspace"
cd /workspace

# Клонируем ваш репо с конфигами
if [ -d "runpod-setup" ]; then
    echo "Репо уже существует, обновляем..."
    cd runpod-setup
    git pull
    cd /workspace
else
    git clone https://github.com/kirillkoks999-oss/runpod-setup.git
fi

# Копируем файлы в workspace
cp -r runpod-setup/* /workspace/
chmod +x /workspace/RunPod_Install.sh

echo "=========================================="
echo "ШАГ 2: Запуск RunPod_Install.sh"
echo "=========================================="
./RunPod_Install.sh

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
pip install -q gradio huggingface_hub hf_transfer hf_xet
echo "✅ Python библиотеки установлены"

echo "=========================================="
echo "ШАГ 6: Скачивание моделей"
echo "=========================================="

export HUGGING_FACE_HUB_TOKEN="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"

# Создаем папки для моделей
mkdir -p /workspace/ComfyUI/models/checkpoints
mkdir -p /workspace/ComfyUI/models/loras

# Пример 1: Скачивание checkpoint с HuggingFace
echo "Скачиваем lily_model.safetensors..."
cd /workspace/ComfyUI/models/checkpoints/
if [ ! -f "lily_model.safetensors" ]; then
    wget --header="Authorization: Bearer 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" \
      https://huggingface.co/rillky/Lily/resolve/main/lily_model.safetensors
    echo "✅ lily_model.safetensors скачан"
else
    echo "lily_model.safetensors уже существует"
fi

# Пример 2: Скачивание LoRA с Civitai
echo "Скачиваем LoRA..."
cd /workspace/ComfyUI/models/loras/
if [ ! -f "example_lora.safetensors" ]; then
    wget -O example_lora.safetensors \
      "https://civitai.com/api/download/models/801399?type=Model&format=SafeTensor"
    echo "✅ LoRA скачана"
else
    echo "LoRA уже существует"
fi

# ДОБАВЬТЕ ЗДЕСЬ СВОИ МОДЕЛИ:
# echo "Скачиваем вашу_модель..."
# cd /workspace/ComfyUI/models/checkpoints/
# wget --header="Authorization: Bearer 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" \
#   https://huggingface.co/автор/модель/resolve/main/файл.safetensors

echo "=========================================="
echo "ШАГ 7: Запуск сервисов"
echo "=========================================="

# Запуск SwarmUI в фоне
echo "Запускаем SwarmUI..."
cd /workspace/SwarmUI
nohup ./launch-linux.sh --launch_mode none --cloudflared-path cloudflared --port 7861 > /workspace/swarm.log 2>&1 &
echo "✅ SwarmUI запущен (логи: /workspace/swarm.log)"
echo "Ждем 10 секунд для инициализации..."
sleep 10

# Запуск Gradio приложения
echo "Запускаем Gradio приложение..."
cd /workspace
export HUGGING_FACE_HUB_TOKEN="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"
python -W ignore Downloader_Gradio_App.py --share

echo "=========================================="
echo "✅ ВСЕ ГОТОВО!"
echo "=========================================="
echo "SwarmUI: проверьте /workspace/swarm.log"
echo "Gradio: должен запуститься выше"
echo "=========================================="