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


```bash
yt-dlp https://www.youtube.com/watch?v=MKd59yzfJKw -k
mv "02. Обучение с подкреплением (25.01.25) [MKd59yzfJKw].mkv" "RL_lect2.mkv"
# ffmpeg -hwaccel cuda -i "testing_video.mkv" -c:v libvpx-vp9 -b:v 500k -vf "scale=1280:720" -c:a aac -b:a 64k "testing_video.webm"
# ffmpeg -i "RL_lect2.mkv" -c:v libvpx-vp9 -b:v 500k -vf "scale=1280:720" -c:a libvorbus -b:a 64k "RL_lect2.webm"


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

```bash
python3 slides/extract_slides2.py -v RL_lect2.mkv -o _output_slides_RL2 -f frame_files_RL2.txt -t 1:31:00
```

## 4. Извлечение аудиодорожки и транскрибация аудио

# create audio track
```bash
ffmpeg -i testing_video_manual.webm -vn -acodec pcm_s16le -ar 16000 -ac 1 testing_video_manual.wav
```

# create audio track
```bash
ffmpeg -i RL_lect2.webm -vn -acodec pcm_s16le -ar 16000 -ac 1 RL_lect2.wav
```

## 5. Транскрибация аудио
```bash
python3 ASR/simple_asr.py --input_file testing_video_manual.wav --return_timestamps 0  --output_file transcription.txt --batch_size 1 --cpu_cores 1
```

```bash
python3 ASR/simple_asr3.py --input_file RL_lect2.wav --return_timestamps 1  --output_file transcription_RL.txt --batch_size 1 --cpu_cores 1 --max_time 1:31:00
```

## 6. Привязывание танскрибации к кадрам
```bash
python3 ASR/assign_slides2.py --transcript_file transcription_RL.txt  --images_folder _output_slides_RL2 --output_json img_text_RL.json --video_file RL_lect2.mkv
```

## 7. Быстрая чистка танскрибации
```bash
python3 ASR/asr_cleanup.py --input_json img_text_RL.json --output_json img_text_RL_adj.json
```

## 8. Быстрая очистка кадров
```bash
python3 vlm/classify_slides.py --input_folder _output_slides_RL2 --output_classification_file RL_images_check.txt --output_captions_file RL_images_captions.txt
```

## 9. Очистка повторяющихся кадров
```bash
python3 vlm/merge_dupl2.py --images_folder _output_slides_RL2 --output_file unique_img_RL.txt # to add an option to use last version of an image
```

## 10. Первая версия аутпута
```bash
python3 output/create_md3.py --json_file img_text_RL_adj.json --images_folder _output_slides_RL2 --unique_img_file unique_img_RL.txt --output_md RL_md1.md --output_transcript_json RL_json1.json --group_duplicates --start_cut 1 --end_cut 2 
```

## 11. Mistral OCR

### Create pdf from all images

```bash
python3 vlm/create_pdf.py --images_folder _output_slides_RL2 --img_class_file unique_img_RL.txt --output_file RL_slides.pdf
```

### Run OCR and save md files

```bash
python3 vlm/mistral_ocr.py --pdf_file RL_slides.pdf --many_pages --output RL_slides_mds
```

### Save terms from pdf file

```bash
python3 vlm/mistral_terms.py --pdf_file RL_slides.pdf --output_txt RL_terms.txt
```



### Run OCR on book chapter and save as md file

```bash
python3 vlm/mistral_ocr.py --pdf_file _pdf/RL_ch2.pdf --output _book/RL_ch2.md
```