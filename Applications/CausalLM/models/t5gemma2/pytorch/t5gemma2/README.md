---
license: gemma
library_name: transformers
pipeline_tag: image-text-to-text 
extra_gated_heading: Access Gemma on Hugging Face
extra_gated_prompt: To access Gemma on Hugging Face, you’re required to review and
  agree to Google’s usage license. To do this, please ensure you’re logged in to Hugging
  Face and click below. Requests are processed immediately.
extra_gated_button_content: Acknowledge license
---

> [!Note]
> This repository corresponds to T5Gemma 2 (pretrained) with 270M encoder and 270M decoder (adapted using UL2)

# T5Gemma 2 model card

**Model Page**: [T5Gemma 2](https://ai.google.dev/gemma/docs/t5gemma2)  

**Resources and Technical Documentation**:

-  [Responsible Generative AI Toolkit](https://ai.google.dev/responsible)
-  [T5Gemma 2 on Kaggle](https://www.kaggle.com/models/google/t5gemma-2)
-  [T5Gemma 2 on Vertex Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/t5gemma)

**Terms of Use**: [Terms](https://ai.google.dev/gemma/terms)  

**Authors**: Google DeepMind


## Model Information

Summary description and brief definition of inputs and outputs.

### Description

T5Gemma is a family of lightweight yet powerful encoder-decoder research models from Google, built by adapting pretrained decoder-only Gemma models into encoder-decoder ones. T5Gemma 2 models, based on Gemma 3, are multilingual and multimodal, handling text and image input and generating text output, with open weights for three pretrained sizes (270M-270M, 1B-1B, and 4B-4B).  

T5Gemma 2 has promising long-context capability, supporting a 128K context window in over 140 languages. Different from T5Gemma, this series adopts tied word embeddings (over encoder and decoder) and merged decoder self- and cross-attention to save model parameters. T5Gemma 2 models are well-suited for a variety of text generation and image understanding tasks, including question answering, summarization, and reasoning. Their relatively small size makes it possible to deploy them in environments with limited resources such as laptops, desktops or your own cloud infrastructure, democratizing access to state of the art AI models and helping foster innovation for everyone.

### Usage

Below we share some code snippets on how to get quickly started with running the model. First, install the Transformers library with:
```sh
pip install -U transformers
```

Then, copy the snippet from the section that is relevant for your usecase.

#### Running with the `pipeline` API

```python
from transformers import pipeline

generator = pipeline(
    "image-text-to-text",
    model="google/t5gemma-2-270m-270m",
)

generator(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
    text="<start_of_image> in this image, there is",
    generate_kwargs={"do_sample": False, "max_new_tokens": 50},
)
```

#### Running the model on a single / multi GPU

```python
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

processor = AutoProcessor.from_pretrained("google/t5gemma-2-270m-270m")
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5gemma-2-270m-270m")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "<start_of_image> in this image, there is"

model_inputs = processor(text=prompt, images=image, return_tensors="pt")
generation = model.generate(**model_inputs, max_new_tokens=20, do_sample=False)
print(processor.decode(generation[0]))
```




### Inputs and outputs

-  **Input:** 
    -  Text string, such as a question, a prompt, or a document to be summarized
    -  Images, normalized to 896 x 896 resolution and encoded to 256 tokens each
    -  Total input context of 128K tokens

-  **Output:** 
    -  Generated text in response to the input, such as an answer to a question, analysis of image content, or a summary of a document
    -  Total output context up to 32K tokens

### Citation

```none
@article{t5gemma2_2025,
  title={T5Gemma 2: Seeing, Reading, and Understanding Longer.},
  author={Zhang, Biao and Suganthan, Paul and Liu, Ga{\"e}l and Philippov, Ilya and Dua, Sahil and Hora, Ben and Black, Kat and Martins, Gus and Sanseviero, Omar and Pathak, Shreya and Hardin, Cassidy and Visin, Francesco and Zhang, Jiageng and Kenealy, Kathleen and Yin, Qin and Lacombe, Olivier and Joulin, Armand and Warkentin, Tris and Roberts, Adam},
  year={2025}
}
```

## Model Data

Data used for model training and how the data was processed.

### Training Dataset

These models were trained on a dataset of text data that includes a wide variety of sources, totalling 2 trillion tokens. The knowledge cutoff date for the training data was August 2024. Here are the key components:

-  Web Documents: A diverse collection of web text ensures the model is exposed to a broad range of linguistic styles, topics, and vocabulary. The training dataset includes content in over 140 languages.
-  Code: Exposing the model to code helps it to learn the syntax and patterns of programming languages, which improves its ability to generate code and understand code-related questions.
-  Mathematics: Training on mathematical text helps the model learn logical reasoning, symbolic representation, and to address mathematical queries.
-  Images: A wide range of images enables the model to perform image analysis and visual data extraction tasks.

The combination of these diverse data sources is crucial for training a powerful multimodal model that can handle a wide variety of different tasks and data formats.

### Data Preprocessing

Here are the key data cleaning and filtering methods applied to the training data:

-  CSAM Filtering: Rigorous CSAM (Child Sexual Abuse Material) filtering was applied at multiple stages in the data preparation process to ensure the exclusion of harmful and illegal content.
-  Sensitive Data Filtering: As part of making Gemma pre-trained models safe and reliable, automated techniques were used to filter out certain personal information and other sensitive data from training sets.
-  Additional methods: Filtering based on content quality and safety in line with [our policies](https://ai.google/static/documents/ai-responsibility-update-published-february-2025.pdf).

## Implementation Information

Details about the model internals.

### Hardware

T5Gemma was trained using [Tensor Processing Unit (TPU)](https://cloud.google.com/tpu/docs/intro-to-tpu) hardware (TPUv4p, TPUv5p and TPUv5e). Training vision-language models (VLMs) requires significant computational power. TPUs, designed specifically for matrix operations common in machine learning, offer several advantages in this domain:

-  Performance: TPUs are specifically designed to handle the massive computations involved in training VLMs. They can speed up training considerably compared to CPUs.
-  Memory: TPUs often come with large amounts of high-bandwidth memory, allowing for the handling of large models and batch sizes during training. This can lead to better model quality.
-  Scalability: TPU Pods (large clusters of TPUs) provide a scalable solution for handling the growing complexity of large foundation models. You can distribute training across multiple TPU devices for faster and more efficient processing.
-  Cost-effectiveness: In many scenarios, TPUs can provide a more cost-effective solution for training large models compared to CPU-based infrastructure, especially when considering the time and resources saved due to faster training.
-  These advantages are aligned with [Google's commitments to operate sustainably](https://sustainability.google/operating-sustainably/).

### Software

Training was done using [JAX](https://github.com/jax-ml/jax) and [ML Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/). JAX allows researchers to take advantage of the latest generation of hardware, including TPUs, for faster and more efficient training of large models. ML Pathways is Google's latest effort to build artificially intelligent systems capable of generalizing across multiple tasks. This is specially suitable for foundation models, including large language models like these ones.  
Together, JAX and ML Pathways are used as described in the [paper about the Gemini family of models](https://goo.gle/gemma2report); _"the 'single controller' programming model of Jax and Pathways allows a single Python process to orchestrate the entire training run, dramatically simplifying the development workflow."_

## Evaluation

Model evaluation metrics and results.

### Benchmark Results

These models were evaluated against a large collection of different datasets and metrics to cover different aspects of text generation. Evaluation results are for pre-trained models. 

#### Reasoning and factuality

| **Benchmark**    | **n-shot**  | **270M-270M**    | **1B-1B**  | **4B-4B**  |
| ---------------- | ----------- | ---------------- | ---------- | ---------- |
| [HellaSwag]      | 10-shot     | 41.1             | 62.9       | 77.4       |
| [BoolQ]          | 0-shot      | 57.4             | 68.5       | 79.3       |
| [PIQA]           | 0-shot      | 66.5             | 74.3       | 79.2       |
| [SocialIQA]      | 0-shot      | 46.1             | 47.7       | 49.9       |
| [TriviaQA]       | 5-shot      | 14.8             | 29.0       | 53.1       |
| [Natural Questions]         | 5-shot      | 3.0              | 8.9        | 17.2       |
| [ARC-c]          | 25-shot     | 27.9             | 41.0       | 56.1       |
| [ARC-e]          | 0-shot      | 57.9             | 67.7       | 74.3       |
| [WinoGrande]     | 5-shot      | 53.9             | 60.8       | 71.6       |
| [BIG-Bench Hard]       | few-shot    | 22.9             | 28.5       | 43.7       |
| [DROP]           | 1-shot      | 39.4             | 51.2       | 66.7       |

[HellaSwag]: https://arxiv.org/abs/1905.07830
[BoolQ]: https://arxiv.org/abs/1905.10044
[PIQA]: https://arxiv.org/abs/1911.11641
[SocialIQA]: https://arxiv.org/abs/1904.09728
[TriviaQA]: https://arxiv.org/abs/1705.03551
[Natural Questions]: https://github.com/google-research-datasets/natural-questions
[ARC-c]: https://arxiv.org/abs/1911.01547
[ARC-e]: https://arxiv.org/abs/1911.01547
[WinoGrande]: https://arxiv.org/abs/1907.10641
[BIG-Bench Hard]: https://paperswithcode.com/dataset/bbh
[DROP]: https://arxiv.org/abs/1903.00161

#### STEM and code

| **Benchmark**    | **n-shot**  | **270M-270M**    | **1B-1B**  | **4B-4B**  |
| ---------------- | ----------- | ---------------- | ---------- | ---------- |
| [MMLU] (Pro CoT)      | 5-shot      | 10.6             | 16.1       | 33.2       |
| [AGIEval]        | 3-5-shot    | 20.9             | 23.8       | 41.1       |
| [MATH]           | 4-shot      | 1.5              | 4.5        | 21.2       |
| [GSM8K]          | 8-shot      | 1.7              | 9.1        | 44.0       |
| [GPQA]           | 5-shot      | 9.6              | 10.5       | 17.6       |
| [MBPP]           | 3-shot      | 5.6              | 25.2       | 44.8       |
| [HumanEval]      | 0-shot      | 4.3              | 15.2       | 30.5       |

[MMLU]: https://arxiv.org/abs/2009.03300
[AGIEval]: https://arxiv.org/abs/2304.06364
[MATH]: https://arxiv.org/abs/2103.03874
[GSM8K]: https://arxiv.org/abs/2110.14168
[GPQA]: https://arxiv.org/abs/2311.12022
[MBPP]: https://arxiv.org/abs/2108.07732
[HumanEval]: https://arxiv.org/abs/2107.03374

#### Multilingual

| **Benchmark**                    | **270M-270M**  | **1B-1B**  | **4B-4B**  |
| -------------------------------- | -------------- | ---------- | ---------- |
| [MGSM]                           | 1.8            | 8.9        | 42.1       |
| [Global-MMLU-Lite]               | 23.5           | 33.1       | 53.5       |
| [WMT24++] (ChrF)                 | 26.9           | 40.9       | 49.2       |
| [FloRes]                         | 23.9           | 33.8       | 41.8       |
| [XQuAD] (all)                    | 40.8           | 63.1       | 70.6       |

[MGSM]: https://arxiv.org/abs/2210.03057
[Global-MMLU-Lite]: https://huggingface.co/datasets/CohereForAI/Global-MMLU-Lite
[WMT24++]: https://arxiv.org/abs/2502.12404v1
[FloRes]: https://arxiv.org/abs/2106.03193
[XQuAD]: https://arxiv.org/abs/1910.11856v3

#### Multimodal capabilities

| **Benchmark**                    | **270M-270M**  | **1B-1B**  | **4B-4B**  |
| -------------------------------- | -------------- | ---------- | ---------- |
| [COCOcap]                        | 69.6           | 86.0       | 105.4      |
| [DocVQA] (val)                   | 41.6           | 66.6       | 74.7       |
| [InfoVQA] (val)                  | 20.2           | 36.4       | 46.0       |
| [MMMU] (pt)                      | 22.7           | 28.4       | 39.4       |
| [TextVQA] (val)                  | 34.1           | 53.1       | 58.4       |
| [RealWorldQA]                    | 27.2           | 42.4       | 46.1       |
| [AI2D]                           | 26.5           | 44.8       | 61.6       |
| [ChartQA]                        | 29.2           | 50.2       | 66.0       |
| [VQAv2]                          | 38.8           | 57.8       | 62.7       |
| [TallyQA]                        | 26.4           | 32.2       | 39.6       |
| [SpatialSense VQA]               | 50.4           | 50.2       | 51.7       |

[COCOcap]: https://cocodataset.org/#home
[DocVQA]: https://www.docvqa.org/
[InfoVQA]: https://arxiv.org/abs/2104.12756
[MMMU]: https://arxiv.org/abs/2311.16502
[TextVQA]: https://textvqa.org/
[RealWorldQA]: https://paperswithcode.com/dataset/realworldqa
[AI2D]: https://allenai.org/data/diagrams
[ChartQA]: https://arxiv.org/abs/2203.10244
[VQAv2]: https://visualqa.org/index.html
[TallyQA]: https://arxiv.org/abs/1810.12440
[SpatialSense VQA]: https://arxiv.org/abs/1908.02660

#### Long-context capabilities

| **Benchmark**  | **270M-270M**  | **1B-1B**  | **4B-4B**                    |
| -------------- | -------------- | ---------- | ---------------------------- |
| [Ruler 32K]    | 57.3           | 69.2       | 81.7                         |
| [Ruler 128K]   | 25.5           | 35.1       | 57.6                         |
| [MRCR 32K]     | 23.4           | 38.6       | 49.4                         |
| [MRCR 128K]    | 20.4           | 32.5       | 39.8                         |

[Ruler 32K]: https://arxiv.org/abs/2404.06654
[Ruler 128K]: https://arxiv.org/abs/2404.06654
[MRCR 32K]: https://arxiv.org/abs/2409.12640
[MRCR 128K]: https://arxiv.org/abs/2409.12640

## Ethics and Safety

Ethics and safety evaluation approach and results.  

**Note: Developers using this model are responsible for the safety of model and derived model outputs.**

### Evaluation Approach

Our evaluation methods include structured evaluations and internal red-teaming testing of relevant content policies. Red-teaming was conducted by a number of different teams, each with different goals and human evaluation metrics. These models were evaluated against a number of different categories relevant to ethics and safety, including:

-  **Child Safety**: Evaluation of text-to-text and image to text prompts covering child safety policies, including child sexual abuse and exploitation.
-  **Content Safety:** Evaluation of text-to-text and image to text prompts covering safety policies including, harassment, violence and gore, and hate speech.
-  **Representational Harms**: Evaluation of text-to-text and image to text prompts covering safety policies including bias, stereotyping, and harmful associations or inaccuracies.

### Evaluation Results

For all areas of safety testing, we saw major improvements in the categories of child safety, content safety, and representational harms relative to previous Gemma models. All testing was conducted without safety filters to evaluate the model capabilities and behaviors. For both text-to-text and image-to-text, and across all model sizes, the model produced minimal policy violations, and showed significant improvements over previous Gemma models' performance with respect to ungrounded inferences. A limitation of our evaluations was they included only English language prompts.    

## Usage and Limitations

These models have certain limitations that users should be aware of.

### Intended Usage

Open vision-language models (VLMs) models have a wide range of applications across various industries and domains. The following list of potential uses is not comprehensive. The purpose of this list is to provide contextual information about the possible use-cases that the model creators considered as part of model training and development.

-  Research and Education
    -  Natural Language Processing (NLP) and VLM Research: These models can serve as a foundation for researchers to experiment with VLM and NLP techniques, develop algorithms, and contribute to the advancement of the field.

### Limitations

-  Training Data
    -  The quality and diversity of the training data significantly influence the model's capabilities. Biases or gaps in the training data can lead to limitations in the model's responses.
    -  The scope of the training dataset determines the subject areas the model can handle effectively.

-  Context and Task Complexity
    -  Models are better at tasks that can be framed with clear prompts and instructions. Open-ended or highly complex tasks might be challenging.
    -  A model's performance can be influenced by the amount of context provided (longer context generally leads to better outputs, up to a certain point).

-  Language Ambiguity and Nuance
    -  Natural language is inherently complex. Models might struggle to grasp subtle nuances, sarcasm, or figurative language.

-  Factual Accuracy
    -  Models generate responses based on information they learned from their training datasets, but they are not knowledge bases. They may generate incorrect or outdated factual statements.

-  Common Sense
    -  Models rely on statistical patterns in language. They might lack the ability to apply common sense reasoning in certain situations.

### Ethical Considerations and Risks

The development of vision-language models (VLMs) raises several ethical concerns. In creating an open model, we have carefully considered the following:

-  Bias and Fairness
    -  VLMs trained on large-scale, real-world text and image data can reflect socio-cultural biases embedded in the training material. These models underwent careful scrutiny, input data pre-processing described and posterior evaluations reported in this card.

-  Misinformation and Misuse
    -  VLMs can be misused to generate text that is false, misleading, or harmful.
    -  Guidelines are provided for responsible use with the model, see the [Responsible Generative AI Toolkit](https://ai.google.dev/responsible).

-  Transparency and Accountability:
    -  This model card summarizes details on the models' architecture, capabilities, limitations, and evaluation processes.
    -  A responsibly developed open model offers the opportunity to share innovation by making VLM technology accessible to developers and researchers across the AI ecosystem.

Risks identified and mitigations:

-  **Perpetuation of biases**: It's encouraged to perform continuous monitoring (using evaluation metrics, human review) and the exploration of de-biasing techniques during model training, fine-tuning, and other use cases.
-  **Generation of harmful content**: Mechanisms and guidelines for content safety are essential. Developers are encouraged to exercise caution and implement appropriate content safety safeguards based on their specific product policies and application use cases.
-  **Misuse for malicious purposes**: Technical limitations and developer and end-user education can help mitigate against malicious applications of VLMs. Educational resources and reporting mechanisms for users to flag misuse are provided. Prohibited uses of T5Gemma models are outlined in the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
-  **Privacy violations**: Models were trained on data filtered for removal of certain personal information and other sensitive data. Developers are encouraged to adhere to privacy regulations with privacy-preserving techniques.

### Benefits

At the time of release, this family of models provides high-performance open vision-language encoder-decoder model implementations designed from the ground up for responsible AI development compared to similarly sized models.  
Using the benchmark evaluation metrics described in this document, these models have shown to provide superior performance to other, comparably-sized open model alternatives.

