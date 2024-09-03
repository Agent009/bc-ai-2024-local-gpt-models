# Encode_AI-and-GPT-Bootcamp-Q3-2024


## Instructions / README
* [Lesson-09 / exercises / 00-Install-Transformers.md](https://github.com/Encode-Club-AI-Bootcamp/Generative-AI-Applications/blob/main/Lesson-09/exercises/00-Install-Transformers.md)
* [Lesson-09 / exercises / 01-Sentiment-Analysis-Example.md](https://github.com/Encode-Club-AI-Bootcamp/Generative-AI-Applications/blob/main/Lesson-09/exercises/01-Sentiment-Analysis-Example.md)
* [Lesson-09 / exercises / 02-Text-Generation.md](https://github.com/Encode-Club-AI-Bootcamp/Generative-AI-Applications/blob/main/Lesson-09/exercises/02-Text-Generation.md)
* [Lesson-09 / exercises / 03-GPT-Text-Generation-Extended.md](https://github.com/Encode-Club-AI-Bootcamp/Generative-AI-Applications/blob/main/Lesson-09/exercises/03-GPT-Text-Generation-Extended.md)
* [Lesson-09](https://github.com/Encode-Club-AI-Bootcamp/Generative-AI-Applications/tree/main/Lesson-09)
* [Lesson-11](https://github.com/Encode-Club-AI-Bootcamp/Generative-AI-Applications/tree/main/Lesson-11)
* [Lesson-11 / exercises / 04-Using-API.md](https://github.com/Encode-Club-AI-Bootcamp/Generative-AI-Applications/tree/main/Lesson-11/exercises/04-Using-API.md)

## Setup

Install one of the Machine Learning frameworks compatible with transformers:

```bash
python.exe -m pip install transformers[torch]
python.exe -m pip install bitsandbytes optimum accelerate
python.exe -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
```

If you get isues with the `fbgemm.dll`, try the fix from [here](https://discuss.pytorch.org/t/failed-to-import-pytorch-fbgemm-dll-or-one-of-its-dependencies-is-missing/201969/13).
Download the [DLL](https://www.dllme.com/dll/files/libomp140_x86_64/00637fe34a6043031c9ae4c6cf0a891d/download) and put it into your `C:\Windows\System32` folder.

Copy `.env.sample` and create your `.env` file, replacing placeholder values with actual values.

## Running
Either run via VSCode / PyCharm / your IDE of choice, or use the command line.

For example, on windows, point to your `venv` and then run your file:

```bash
& /path/to/python.exe /path/to/script.py
```

## Resources
* [GitHub Repo - Encode-Club-AI-Bootcamp / Generative-AI-Applications](https://github.com/Encode-Club-AI-Bootcamp/Generative-AI-Applications)