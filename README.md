# BioAMLA - Bioacoustics & Machine Learning Applications

The BIOAMLA project is a collection of resources for introductory topics in bioacoustics, sound processing, and machine learning. Supplementary code examples are being developed to demonstrate bioacoustics and machine learning techniques using Python / Jupyter notebooks. These samples will run best on a GPU equipped device. Code has been tested on Ubuntu 22.04.5 LTS and Python 3.12 using Nvidia GTX 1660Ti 6GB, RTX 4060 Ti 8GB, and RTX 5060 Ti 16GB GPUs.

## Setup

To run the notebooks or apps, you'll need to set up your Python environment:

1. Clone this repository:

   ```bash
   git clone https://github.com/jmcmeen/bioamla-hub.git
   cd bioamla-hub
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter to run notebooks:

   ```bash
   jupyter notebook
   ```

For GPU support, ensure you have the appropriate CUDA drivers and PyTorch GPU version installed.

## Notebooks

- [Using PyINaturlist To Download Audio from La Selva Biological Station, Costa Rica](notebooks/inaturalist_la_selva.ipynb)
- [Wave File Dissection (Basics)](notebooks/dissecting_wav_files.ipynb)
- [Testing GPU Capabilities for PyTorch](notebooks/torch_gpu_test.ipynb)

## Apps

- [Streamlit App for Audio Classification Using AST](/apps/ast_audio_classifier/)

## BIOMLA Projects

- [jmcmeen/frogs-of-steele-creek](https://github.com/jmcmeen/frogs-of-steele-creek): The Frogs of Steele Creek Park is an introduction to bioacoustics, computer science, and machine learning, featuring research from Appalachian Mountains.
- [jmcmeen/bioamla-datasets](https://github.com/jmcmeen/bioamla-datasets): BIOAMLA Datasets is a collection of raw and annotated datasets created by BIOAMLA for current research in the Appalachian Mountains and Costa Rica.

## Related Bioacoustics Software

The bioacoustics community has developed a rich ecosystem of open-source tools for wildlife sound analysis. Below are notable projects.

- [BirdNET-Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer) - Batch processing of large audio corpora with species detections
- [BirdNET-Go](https://github.com/tphakala/birdnet-go) - Real-time BirdNET-based monitoring with dashboards
- [HawkEars](https://github.com/jhuus/HawkEars) - Desktop scanner for Canadian birds, outputs Audacity label files
- [OpenSoundscape](https://github.com/kitzeslab/opensoundscape) - Python library for preprocessing and training CNNs with PyTorch
- [scikit-maad](https://github.com/scikit-maad/scikit-maad) - Quantitative soundscape analysis and acoustic indices
- [BirdNET (Python)](https://github.com/birdnet-team/birdnet) - Programmatic BirdNET species prediction API
- [PyHa](https://github.com/UCSD-E4E/PyHa) - Convert weak labels to strong intra-clip labels
- [bambird](https://github.com/ear-team/bambird) - Reduce label noise in bird sound datasets
- [Bioacoustics Datasets](https://github.com/bioacoustic-ai/bioacoustics-datasets) - Curated list of datasets for deep learning
- [Perch](https://github.com/google-research/perch) - Pretrained bioacoustics classifier with embeddings (Apache-2.0 license)
- [Perch Hoplite](https://github.com/google-research/perch-hoplite) - Embedding storage and retrieval workflows for active learning
- [BirdVoxDetect](https://github.com/BirdVox/birdvoxdetect) - Deep learning system for nocturnal flight call detection
- [BEANS](https://github.com/earthspecies/beans) - Standardized bioacoustics evaluation across multiple datasets and tasks
- [BirdSet](https://github.com/DBD-research-group/BirdSet) - Large-scale avian benchmark dataset for reproducible research
- [Acoustic Representation Toolbox](https://github.com/earthspecies/acoustic-representation-toolbox) - Earth Species work in acoustic representations
- [Vesper](https://github.com/HaroldMills/Vesper) - Software for acoustic wildlife monitoring, especially nocturnal flight calls
- [Bioacoustics Model Zoo](https://github.com/kitzeslab/bioacoustics-model-zoo) - Collection of pretrained bioacoustics models
- [Microsoft Akoustos](https://github.com/microsoft/Akoustos) - Cloud-scale bioacoustic analysis platform

## Related Organizations

- [K. Lisa Yang Center for Conservation Bioacoustics](https://www.birds.cornell.edu/ccb/) - Cornell Lab research center advancing bioacoustics science and technology
- [Macaulay Library](https://www.macaulaylibrary.org/) - World's largest archive of wildlife sounds and videos
- [iNaturalist](https://www.inaturalist.org/) - Citizen science platform for biodiversity observations with AI-powered species identification
- [xeno-canto](https://xeno-canto.org/) - Community-driven platform for sharing bird sounds worldwide
- [Earth Species Project](https://www.earthspecies.org/) - Nonprofit using AI to decode non-human communication
- [Rainforest Connection](https://www.rfcx.org/) - Real-time acoustic monitoring for rainforest protection
- [International Bioacoustics Council (IBAC)](https://www.ibac.info/) - Professional society promoting bioacoustics research
- [Bioacoustic Unit, University of Alberta](https://bioacoustic.abmi.ca/) - Research unit focused on acoustic monitoring of biodiversity
- [Borror Laboratory of Bioacoustics](https://borror.osu.edu/) - Ohio State University research lab and sound archive
- [British Library Sound Archive](https://www.bl.uk/subjects/sound) - Extensive collection of wildlife and environmental recordings
- [Orcasound](https://www.orcasound.net/) - Open-source hydrophone network for orca conservation
- [Wild Me](https://www.wildme.org/) - AI-powered wildlife identification and population monitoring

## Bioacoustics Resources on Hugging Face 🤗

- [NatureLM-audio](https://huggingface.co/EarthSpeciesProject/NatureLM-audio) - First audio-language foundation model for bioacoustics (Earth Species Project)
- [Perch](https://huggingface.co/cgeorgiaw/Perch) - Google's Perch model for embeddings and classification (~15k species)
- [WhisperSeg Animal VAD](https://huggingface.co/nccratliri/whisperseg-base-animal-vad) - Voice Activity Detection for human vs. animal vocalizations
- [AudioProtoPNet BirdSet](https://huggingface.co/DBD-research-group/AudioProtoPNet-5-BirdSet-XCL) - Interpretable model trained on BirdSet
- [Wav2Vec2 BirdSet](https://huggingface.co/DBD-research-group/Wav2Vec2-Base-BirdSet) - Wav2Vec2 pretrained on BirdSet corpus
- [AST Bird Voice](https://huggingface.co/JamesStratford/ast-finetuned-voice-of-birds) - Audio Spectrogram Transformer for bird vocalizations
- [WhAM](https://huggingface.co/Project-CETI/wham) - Whale Acoustics Model for sperm whale codas
- [Animal Sounds Wav2Vec2](https://huggingface.co/ardneebwar/wav2vec2-animal-sounds-finetuned-hubert-finetuned-animals) - General animal sounds classification
- [BirdSet](https://huggingface.co/datasets/DBD-research-group/BirdSet) - Major benchmark for avian bioacoustics
- [BEANS-Zero](https://huggingface.co/datasets/EarthSpeciesProject/BEANS-Zero) - Zero-shot benchmark for bioacoustic foundation models
- [NatureLM Training Data](https://huggingface.co/datasets/EarthSpeciesProject/NatureLM-audio-training) - Massive audio-text pairs from Xeno-Canto, Macaulay, and others
- [WABAD](https://huggingface.co/datasets/DBD-research-group/WABAD) - World Annotated Bird Acoustic Dataset
- [WhaleSounds](https://huggingface.co/datasets/monster-monash/WhaleSounds) - Antarctic baleen whale sounds
- [InsectSound](https://huggingface.co/datasets/monster-monash/InsectSound) - Insect vibration/sound recordings
- [DogSpeak](https://huggingface.co/datasets/ArlingtonCL2/DogSpeak_Dataset) - Large-scale canine vocalizations
- [The Wilds PAM](https://huggingface.co/datasets/imageomics/thewilds_bioacousticmonitors) - Raw passive acoustic monitoring data


## Acknowledgments

- Thank you to The Friends of Steele Creek Nature Center and Park for grant support for the 2023 SCP Wetlands project. [Friends of Steele Creek Park](https://www.friendsofsteelecreek.org/)
- Thank you to the staff at The Organization for Tropical Studies and La Selva Research Station for guidance and administrative support while collecting data in 2024. [OTS/OET](https://tropicalstudies.org/)
