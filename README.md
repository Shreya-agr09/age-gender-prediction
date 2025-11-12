# Age and Gender Prediction
 ## Overview

This project was part of the Deep Learning & GenAI (NPPE1) Kaggle Competition, where the task was to predict a person’s age (regression) and gender (binary classification) from facial images.
The competition aimed to evaluate skills in model design, training, fine-tuning, and deployment.

## 🧩 Objective

Develop a PyTorch-based model capable of predicting:

Age: Numeric regression value

Gender: Binary class (0 = Female, 1 = Male)

## 📦 Dataset

We are provided:

Training dataset — Face images with labels (age, gender)

Test dataset — Face images without labels

The goal is to produce a submission.csv with predictions for each test image:

id,age,gender
0001.jpg,23,1
0002.jpg,41,0
...
## Approach

Made a Cnn model from cratch giving score of 0.47

Used Transfer Learning to use pre trainned models

## Model Comparisons
| Model              | Epochs | Age Normalization | Fine-Tuning        | Score | KaggleHub Name                    |
|---------------------|--------|------------------:|--------------------|-------:|-----------------------------------|
| CNN Baseline        | 10     | No                | None               | 0.471 | age-gender-cnn                    |
| EfficientNet-B0     | 10     | No                | Gradual unfreeze   | 0.589 | age-gender-pre-trained-cnn        |
| ConvNeXtV2-Tiny     | 5      | Yes               | Gradual unfreeze   | 0.608 | age-gender-fb-trained-model       |
| ConvNeXtV2-Tiny     | 12     | Yes               | Gradual unfreeze   | 0.611 | age-12gender-con-trained-model    |
| **ConvNeXtV2-Tiny** | **12** | **No**            | **Gradual unfreeze** | **0.798** | **age-12gender_wn-con-trained-model** |
| ResNet-50           | 10     | Yes               | Gradual unfreeze   | 0.594 | resnet50-age-gender               |
| ResNet-50           | 10     | No                | Frozen backbone    | 0.575 | resnet50-v2age-gender             |

## 🔑 Key Observations

Pretrained backbones significantly outperform scratch CNNs.

ConvNeXt-Tiny (without age normalization) achieved the best score (0.798).

Gradual unfreezing improved stability and generalization.

Target normalization helped regression convergence.

## 🧠 Architecture Summary (Scratch CNN)
Conv2d → ReLU → BatchNorm2d → MaxPool2d

Conv2d → ReLU → BatchNorm2d → MaxPool2d

Flatten → FC(128 → 64)

Output Heads: 

   - Age → Linear(64 → 1)
     
   - Gender → Linear(64 → 1)
     
Loss:

   - MSELoss() for Age
     
   - BCEWithLogitsLoss() for Gender
     
Optimizer: Adam (lr=0.001)

Epochs: 10

Score: 0.471

## 🧩 Final Conclusion

ConvNeXt-Tiny is the most effective backbone for joint age–gender prediction.

Techniques like dynamic loss weighting, advanced augmentation, and hyperparameter tuning could further improve results.

Future extensions may include:

Multi-task loss balancing

Attention pooling

Integration with lightweight models for real-time inference

## 🏁 Result

Best Model: ConvNeXtV2-Tiny (12 epochs, no age normalization)

Final Score: 0.798
