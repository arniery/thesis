# Modifying the TTS pipeline and voice conversion models for the accent conversion task
@ Trinity College Dublin CLCS/SLP Programme (2025), Master's Thesis

## summary
ABSTRACT: Traditional text-to-speech systems consist principally of the acoustic model and the vocoder, and systems aiming for voice conversion include the voice conversion model in between those components as well as a speaker embedding and/or dialect accent embedding step. This thesis aims to explore a) how voice/speaker conversion technology can be utilised for the accent conversion task and b) how the text-to-speech pipeline can be modified to include the accent conversion step. Within this paper, both traditional and end-to-end TTS systems are researched and reviewed, the different components and the different open-source options available for each are examined, and two experiments are run in attempts to achieve accent conversion from American accented English to Indian English. It was found that the main voice conversion model available for use today, AutoVC, is challenging to train, but that achieving the desired result would likely be possible with the right materials. The experiment utilising a pretrained acoustic model, voice converter, and vocoder, achieves noticeable but unpolished results. Learnings and advice for further research are outlined after the experiments are completed.

## outline of files
a_b.wav : is an example result from the miniexperiment notebook. if you'd like to try running/producing other examples, you can generate neutral phrases in the "gTTS" section and replace the speech1.npy in path1 with the source file of your choosing.

hparams.py, main.py, make_metadata.py, make_spect.py, model_vc.py, and model_vc_clean.py : edited scripts for the training of the AutoVC voice conversion model using an Apple Silicon (M2) CPU and the IndicTTS mono Hindi dataset (which contains 13 hours of spoken Hindi English data from 2 speakers, a male speaker and a female speaker).
--> license-indictts.pdf : license for the IndicTTS dataset from IIT Madras.

inference2.ipynb : includes the pipeline with Tacotron2 as the acoustic model, our custom-trained with IndicTTS AutoVC model, and the custom-trained HiFi-GAN inference. The custom HiFi-GAN training can be replaced with the pretrained version if desired, and this was the first way that this was tried.

inference3.ipynb : includes gTTS-generated source data, custom HiFi-GAN training, config modification to match AutoVC's config, target embedding extraction, and debugging scripts as well as the full "experiment 1" pipeline. To skip the .npy file load, scroll 2/3rds-3/4ths of the way down the file to view the rest.

miniexperiment.ipynb : contains the pipeline which utilises gTTS to produce its 3 source data files (neutral/American accented English), a pretrained AutoVC model by Nicola Landro, and the corresponding WaveNet vocoder to produce a listenable result for comparison with the expected result from a custom-trained AutoVC model.

Notebooks are completely open for download and further research if you think you can produce a result from experiment 1 (inference 2 and inference3). Don't hesitate to contact me if you do create a listenable audio from this pipeline. Sources for code and research are linked below, and full paper will be linked in future.

## sources
[1] AIology. (2021, October 25). WaveNet (Theory and Implementation) [Video]. YouTube. https://www.youtube.com/watch?v=KCk1i5xRxLA.
[2] Costa, D. (2019). Indian English — A National Model. Journal of English as an International Language, 14(2), 16–28. https://files.eric.ed.gov/fulltext/EJ1244241.pdf.
[3] Dhakal Chhetri, G. Bdr.; Dahal, K. Chandra; Poudyal, P. “Impacts of Vocoder Selection on Tacotron-based Nepali Text-to-Speech Synthesis”. In: Proceedings of the First Workshop on Challenges in Processing South Asian Languages (CHiPSAL 2025) (January 19, 2025), pp. 185–192. International Committee on Computational Linguistics. https://aclanthology.org/2025.chipsal-1.18.pdf
[4] Donahue, J.; Dieleman, S.; Binkowski, M.; Elsen, E.; and Simonyan, K. “End- to-End Adversarial Text-to-Speech”. In: CoRR abs/2006.03575 (2020). arXiv: 2006.03575. URL: https://arxiv.org/abs/2006.03575. 
[5] Drhová, A. (2025). End-to-End Speech Synthesis: A Survey and Basic Implementation. [Master’s Thesis, Czech Technical University Department of Radioelectronics]. ČVUT DSpace. 
[6] Falai, A. (2021). Conditioning text-to-speech synthesis on dialect accent: a case study. [Master’s Thesis, Università di Bologna Department of Computer Science and Engineering]. AMS Tesi di Laurea.
[7] Ghorbani, S.; Hansen,  John H. L.; Advanced accent/dialect identification and accentedness assessment with multi-embedding models and automatic speech recognition. J. Acoust. Soc. Am. 1 June 2024; 155 (6): 3848–3860. https://doi-org.elib.tcd.ie/10.1121/10.0026235
[8] Griffin, D. and Lim, J. “Signal estimation from modified short-time Fourier transform”. In: IEEE Transactions on Acoustics, Speech, and Signal Processing 32.2 (1984), pp. 236–243. DOI: 10.1109/TASSP.1984.1164317. 
[9] Hsu, P. (2022) Tacotron-PyTorch [Source code]. GitHub. https://github.com/BogiHsu/Tacotron2-PyTorch
[10] Hsu, P.; Hung-yi, L. (2024). WG-WaveNet: Real-Time High-Fidelity Speech Synthesis without GPU [Source code]. GitHub. https://github.com/BogiHsu/WG-WaveNet
[11] Ito, K. and Johnson, L. “The LJ Speech Dataset.” https://keithito.com/LJ-Speech-Dataset/, 2017. Cited on August 28, 2025.
[12] Kesharwani, A. (2025). Building an Accent Embedding Model from Scratch: A Step-by-Step Technical Guide [Article]. Medium. https://medium.com/@adarshhme/building-an-accent-embedding-model-from-scratch-a-step-by-step-technical-guide-3d336577fb4a 
[13] Koluguri, Park & Ginsburg for NVIDIA (2021). TitaNet. 
[14] Kong, J.; Kim, J.; Bae, J. “HiFiGAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis”. 2020. In: 34th Conference on Neural Information Processing Systems (NeurIPS 2020), Vancouver, Canada. arXiv: 2010.05646v2.
[15] Kong, J. (2020). HiFiGAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis [Source code]. GitHub. https://github.com/jik876/hifi-gan 
[16] Kumar, K.; Kumar, R.; de Boissiere, T.; Gestin, L.; Teoh, W. Z.; Sotelo, J.; de Bre- bisson, A.; Bengio, Y.; and Courville, A. MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis. 2019. arXiv: 1910.06711 [eess.AS]. 
[17] Landro, N. (2021). AUTOVC [Source code]. https://github.com/nicolalandro/autovc
[18] Li, Y. A.; Zare, A.; and Mesgarani, N. “StarGANv2-VC: A Diverse, Unsupervised, Non-parallel Framework for Natural-Sounding Voice Conversion”. In: CoRR abs/2107.10394 (2021). arXiv: 2107 . 10394. URL: https :
//arxiv.org/abs/2107.10394.
[19] Murthy, H. A. & Umesh, S. (2016). IndicTTS Hindi English Dataset. 2016 TTS Consortium, TDIL, Meity. DEPARTMENT OF Computer Science and Engineering and Electrical Engineering, IIT Madras.
[20]  Nguyen, T. N.; Pham, N. Q.; Waibel, A. (2022). Accent Conversion using Pre-trained Model and Synthesized Data from Voice Conversion. Interspeech 2022 p. 2583-2587. https://isl.iar.kit.edu/downloads/nguyen22d_interspeech.pdf
[21] van den Oord, A; Dieleman, S.; Zen H.; Simonyan K.; Vinyals O.; Graves, A.; Kalchbrenner, N.; Senior, A. W.; and Kavukcuoglu, K. “WaveNet: A Generative Model for Raw Audio”. In: CoRR abs/1609.03499 (2016). arXiv: 1609.03499. URL: http://arxiv.org/abs/1609.03499. 
[22] Qian, K. (2019). AutoVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss [Source code]. GitHub. https://github.com/auspicious3000/autovc
[23] Qian, K.; Zhang, Y.; Chang, S.; Yang, X.; Hasegawa-Johnson, M. (2019). AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss. Thirty-sixth International Conference on Machine Learning (ICML 2019). https://arxiv.org/abs/1905.05879.
[24] Ren, Y.; Ruan, Y.; Tan, X.; Qin, T.; Zhao, S.; Zhao, Z.; and Liu, T. “FastSpeech: Fast, Robust and Controllable Text to Speech”. In: CoRR abs/1905.09263 (2019). arXiv: 1905.09263. URL: http://arxiv.org/abs/1905.09263. 
[25] Shao, W.; Shao, Y.; Li, C. (2024). A novel regularization method for decorrelation learning of non-parallel hyperplanes. In: Elsevier Information Sciences, Volume 667 May 2024, 120461. https://doi.org/10.1016/j.ins.2024.120461
[26] Thomas, K. J. (2024). Angular Softmax loss [Article]. https://knowledge.kevinjosethomas.com/Artificial-Intelligence/Angular-Softmax-Loss
[27] Wang, Y. et al. “Tacotron: A Fully End-to-End Text-To-Speech Synthesis Model”. In: CoRR abs/1703.10135 (2017). arXiv: 1703 . 10135. URL: http://arxiv.org/abs/1703.10135. 
[28] Yamagishi Junichi, C. M. K.; Veaux. CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92). 2019. DOI: https://doi.org/10.7488/ds/2645.
[29] Yamamoto, R. (2020). WaveNet vocoder [Source code]. GitHub. https://github.com/r9y9/wavenet_vocoder
