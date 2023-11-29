# Zero Shot Classification using Huggingface Pipeline

## Overview

This project applies a zero-shot learning approach to classify sentences from Yelp restaurant reviews based on key aspects like food, service, and ambiance. A pre-trained natural language inference (NLI) model is leveraged to categorize review text without any category-specific labeled data.

The technique ultimately achieves 82% accuracy in assigning relevant aspect labels. Careful tuning of the classification threshold and crafting an effective hypothesis template for the NLI model led to significant performance gains.

## Data

The dataset consists of 584 sentences extracted from restaurant reviews on Yelp. Each sentence has a ground truth aspect label:

- **Food**: Commenting on the taste, quality, presentation, etc. of the dishes and drinks
- **Service**: Discussing the attentiveness, friendliness, and professionalism of the staff
- **Ambiance**: Describing the overall atmosphere and physical environment of the restaurant
- **Other**: Sentences not covered by the above categories

## Approach

The zero-shot classification pipeline from Hugging Face is leveraged out-of-the-box, using a pre-trained natural language inference (NLI) model called 'facebook/bart-large-mnli'. 

The process involves creating label-specific hypothesis sentences that are paired with the review premise sentences. The model then predicts an entailment probability for each premise-hypothesis pair. The label with the maximum probability is selected as the predicted class.

Tuning the **hypothesis template** to better match the review domain led to significant jumps in accuracy:

- Default: "This example is {label}."  
- Custom: "This review is related to the restaurant's {label}."

The **classification threshold** was also optimized to only assign food/service/ambiance labels when the model was sufficiently confident, relegating unclear cases to an 'other' category.

## Performance

- **Accuracy**: 82%

Carefully crafting the hypothesis template and tuning the decision threshold based on model confidence scores were key to maximizing classification accuracy.

The confusion matrix and classification report provide deeper insight into the performance per category. Food is the best performing class, while ambiance remains challenging.

## Future Work

This zero-shot NLI approach shows promising results without requiring any labeled training data. Next steps could involve:

- Experimenting with model variations and fine-tuning
- Expanding the hypothesis templates for greater specificity
- Growing the unlabeled review dataset 
- Comparing against supervised models

The core technique demonstrates the power of pre-trained NLI models for customizable text classification.
