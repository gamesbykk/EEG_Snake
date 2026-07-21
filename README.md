# EEG Snake
An EEG project that reads brainwaves and predicts the action the user wants to take in the game snake. This is from last year so it probably doesn't work anymore.
I made a Brain controlled snake game using BCI (Brain-Computer Interface ) and EEG. 
## Technical Details
This project implements a brain-controlled version of the classic Snake game using electroencephalography (EEG) signals. An LSTM (Long Short-Term Memory) neural network is trained to classify EEG data into four movement commands: left, right, up, and down, allowing the player to control the snake using brain activity rather than a keyboard or controller.
The system consists of an EEG signal processing pipeline, data preprocessing and feature extraction, an LSTM-based classifier, and a game interface that converts the model's predictions into real-time movement commands. The project demonstrates the application of brain-computer interfaces (BCIs) and deep learning for human-computer interaction.
## Key Challenges
## What I learned and challenges I faced
Developing an EEG-controlled game is challenging because EEG signals are highly noisy, non-stationary, and vary significantly between users. The model must distinguish meaningful patterns in brain activity while remaining robust to artefacts such as eye blinks, muscle movements, and electrical interference. Additionally, predictions must be generated with low latency so that the game remains responsive.
Using an LSTM network is well suited to this problem because EEG signals are time-series data, allowing the model to learn temporal patterns associated with different movement intentions. Achieving reliable real-time performance required careful preprocessing, feature engineering, and tuning of the neural network architecture.
Implemented EEG signal preprocessing and filtering.
Implemented an LSTM model for EEG signal classification.
Mapped predicted classes to real-time Snake game controls.
Integrated the trained model with the game engine for live gameplay.
By completing this project, I gained practical experience with brain-computer interfaces, time-series deep learning, and neural network training. I also learned how to process noisy biomedical signals, optimise sequence models such as LSTMs, and integrate machine learning models into interactive real-time applications. This expanded my understanding of artificial intelligence, signal processing, and human-computer interaction.
## What the Bot can do
Interpret EEG signals and classify them into four movement commands (left, right, up, and down).
Control the Snake game in real time using brain activity.
Generate movement predictions with low enough latency for responsive gameplay.
Maintain reliable performance despite noisy EEG data through preprocessing and filtering.
Demonstrate the feasibility of using deep learning for brain-computer interface applications.
## Similar Existing Projects
Brain-computer interface research has demonstrated that EEG signals can be used to control computers, wheelchairs, robotic arms, and simple games. Projects such as the BrainGate research programme and OpenBCI-based applications have shown the potential of EEG-controlled interfaces, while many academic studies have applied recurrent neural networks to EEG classification tasks. This project follows a similar approach by using an LSTM model to decode EEG signals into directional commands, but focuses on controlling a real-time Snake game as an accessible demonstration of brain-computer interface technology.
