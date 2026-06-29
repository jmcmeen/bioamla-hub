# BIOAMLA Hub

> A curated index of the BIOAMLA ecosystem — tools, datasets, related software, and organizations for researchers getting oriented in bioacoustics and machine learning.

[![bioamla on PyPI](https://img.shields.io/pypi/v/bioamla?label=bioamla%20on%20PyPI)](https://pypi.org/project/bioamla/)

## About

BIOAMLA (Bioacoustics & Machine Learning Applications) is a collection of resources for
bioacoustics, sound processing, and machine learning. This repository is a human-readable
resource hub: a curated index of the BIOAMLA ecosystem's first-party tools and datasets,
along with related community software, organizations, and Hugging Face resources for
researchers getting oriented in the field.

## BIOAMLA Projects

| Repository | Description |
| --- | --- |
| [bioamla](https://github.com/jmcmeen/bioamla) | Core Python library and CLI for bioacoustics and ML |
| [bioamla-datasets](https://github.com/jmcmeen/bioamla-datasets) | Curated environmental audio datasets |
| [bioamla-workflows](https://github.com/jmcmeen/bioamla-workflows) | Kedro and n8n workflow templates |
| [bioamla-hub](https://github.com/jmcmeen/bioamla-hub) | This repository — resource index |
| [magpy](https://github.com/jmcmeen/magpy) | Desktop GUI for spectrograms and annotation (PyQt6) |
| [magpy-lite](https://github.com/jmcmeen/magpy-lite) | Lightweight companion annotation GUI |
| [observa](https://github.com/jmcmeen/observa) | Dockerized iNaturalist Open Data platform |
| [riffy](https://github.com/jmcmeen/riffy) | Zero-dependency RIFF/WAV parsing library |
| [terrana](https://github.com/jmcmeen/terrana) | Zero-config spatial API server |
| [frogs-of-steele-creek](https://github.com/jmcmeen/frogs-of-steele-creek) | Frog species identification for Steele Creek Park |

## Related Bioacoustics Software

The bioacoustics community has developed a rich ecosystem of open-source tools for wildlife
sound analysis. Below are notable projects. _Last reviewed: June 2026._

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

### Models

- [NatureLM-audio](https://huggingface.co/EarthSpeciesProject/NatureLM-audio) - First audio-language foundation model for bioacoustics (Earth Species Project)
- [Perch](https://huggingface.co/cgeorgiaw/Perch) - Google's Perch model for embeddings and classification (~15k species)
- [WhisperSeg Animal VAD](https://huggingface.co/nccratliri/whisperseg-base-animal-vad) - Voice Activity Detection for human vs. animal vocalizations
- [AudioProtoPNet BirdSet](https://huggingface.co/DBD-research-group/AudioProtoPNet-5-BirdSet-XCL) - Interpretable model trained on BirdSet
- [Wav2Vec2 BirdSet](https://huggingface.co/DBD-research-group/Wav2Vec2-Base-BirdSet) - Wav2Vec2 pretrained on BirdSet corpus
- [AST Bird Voice](https://huggingface.co/JamesStratford/ast-finetuned-voice-of-birds) - Audio Spectrogram Transformer for bird vocalizations
- [WhAM](https://huggingface.co/Project-CETI/wham) - Whale Acoustics Model for sperm whale codas
- [Animal Sounds Wav2Vec2](https://huggingface.co/ardneebwar/wav2vec2-animal-sounds-finetuned-hubert-finetuned-animals) - General animal sounds classification

### Datasets

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
