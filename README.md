The objective of the research work is to propose a convolutional neural network (CNN) for modulation classification by generate synthetic, channel-impaired waveforms. Using the generated waveforms as training data, you train a CNN for modulation classification. Then the CNN is tested and validated for accuracy.
The trained CNN in this example recognizes these modulation types: Binary phase shift keying (BPSK), Quadrature phase shift keying (QPSK), 8-ary phase shift keying (8-PSK), 16-ary quadrature amplitude modulation (16-QAM), 64-ary quadrature amplitude modulation (64-QAM), Gaussian frequency shift keying (GFSK)
Generate 10,000 frames for each modulation type, where 80% is used for training, 10% is used for validation and 10% is used for testing. We use training and validation frames during the network training phase. Final classification accuracy is obtained using test frames. Each frame is 1024 samples long and has a sample rate of 200 kHz. For digital modulation types, eight samples represent a symbol. The network makes each decision based on single frames rather than on multiple consecutive frames.
The CNN is designed for Modulation classification with the following parameters and obtained the performance metrics.
1. The no. of layers for this CNN Model is 9.
2. Training duration 168 minutes 34 seconds.
3. The training accuracy reached is 96.65%.
4. The testing accuracy is 96.65% and the network has certain difficulty in detecting 16-QAM and 64-QAM frames. This problem is expected since each frame carries only 128 symbols and 16-QAM is a subset of 64-QAM. The network also confuses QPSK and 8-PSK frames, since the constellations of these modulation types look similar once phase-rotated due to the fading channel and frequency offset.
