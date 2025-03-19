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


## 2. Run OCR on book chapter and save as md file (optional)

```bash
python3 vlm/mistral_ocr.py --pdf_file _pdf/RL_ch2.pdf --output _book/RL_ch2.md
```


## 3. Run main Streamlit app (lecture summary)

```bash
streamlit run app.py
```

## 4. Run a RAG app - upload a pdf or MD created with Mistral (better)

```bash
streamlit run rag/rag_prod2.py -- --terms_list temp/terms.txt
```