# Speech enhancement
Репозиторий с проектом по теме Youtube / Video lecture summarize


 
# Использование кода

```bash
cd lect_summ
```

## 1. Настройка окружения

```bash
conda env create -f environment.yml

```

```bash
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .

```

Если нет ffmpeg, для Ubuntu его нужно собрать (https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu)

```bash
utils/ffmpeg_install.sh
```


## 2. Скачивание и подготовка данных
### Пример YT video
```bash
yt-dlp https://www.youtube.com/watch?v=CSFjlhr_cz0
mv "05. LLM Validation (04.02.25) [CSFjlhr_cz0].mkv" "testing_video.mkv"
# ffmpeg -hwaccel cuda -i "testing_video.mkv" -c:v libvpx-vp9 -b:v 500k -vf "scale=1280:720" -c:a aac -b:a 64k "testing_video.webm"
ffmpeg -i "testing_video.mkv" -c:v libvpx-vp9 -b:v 500k -vf "scale=1280:720" -c:a libvorbus -b:a 64k "testing_video.webm"


```

## 3. Извлечение кадров слайдов из видео

