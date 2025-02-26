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


## 2A. Скачивание и подготовка данных
### Пример YT video
```bash
yt-dlp https://www.youtube.com/watch?v=CSFjlhr_cz0
mv "05. LLM Validation (04.02.25) [CSFjlhr_cz0].mkv" "testing_video.mkv"
# ffmpeg -hwaccel cuda -i "testing_video.mkv" -c:v libvpx-vp9 -b:v 500k -vf "scale=1280:720" -c:a aac -b:a 64k "testing_video.webm"
ffmpeg -i "testing_video.mkv" -c:v libvpx-vp9 -b:v 500k -vf "scale=1280:720" -c:a libvorbus -b:a 64k "testing_video.webm"


```

## 2B. Скачивание готового YT video
### Пример YT video
```bash
python3 utils/video_dl.py https://drive.google.com/file/d/1W6eXoNTQ8gC78327MphUugjhg1KDC16m/view?usp=sharing
```



## 3. Извлечение кадров слайдов из видео
```bash
python3 slides/extract_slides.py -v testing_video_manual.webm -o _output_slides -f frame_files.txt
```

## 4. Извлечение аудиодорожки и транскрибация аудио

# create audio track
```bash
ffmpeg -i testing_video_manual.webm -vn -acodec pcm_s16le -ar 16000 -ac 1 testing_video_manual.wav
```

## 5. Транскрибация аудио
```bash
python3 ASR/simple_asr.py --input_file testing_video_manual.wav --return_timestamps 0  --output_file transcription.txt --batch_size 1 --cpu_cores 1
```



