

Public Artifact:  

- [x] Title
- [x] Released
- [x] Link
    
    assessment-algorithm/
    
- Application/Scenario/Domain of Misuse: Criminal Risk Assessment (Predictive Algorithm)
    
- Regulated Domain/Protected Class: Education/Race
    
- Evidence: Dataset - https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-
    
    data-and-analysis

---

```python
# Import machine learning fairness library
import fairness_ml

# Define function to reweigh training data
def reweigh_data(training_data):

    # Apply reweighing to the training data
    reweighted_data = fairness_ml.reweigh(training_data, 'age')
    return reweighted_data

# Define function to train model with adversarial debiasing
def train_model_with_debiasing(reweighted_data):

    # Initialize adversarial debiasing model
    model = fairness_ml.AdversarialDebiasing()
    
    # Train the model with the reweighted data
    model.fit(reweighted_data)
    
    return model

# Load training data
training_data = load_data('training_dataset.csv')

# Apply reweighing to the training data
reweighted_data = reweigh_data(training_data)

# Train the model with adversarial debiasing
fair_model = train_model_with_debiasing(reweighted_data)

# Predict on new data
new_applicant_data = load_data('new_applicant_dataset.csv')
predictions = fair_model.predict(new_applicant_data)

```
# Question 1

**Finding a Public Artifact (25 points)**:

- **Objective**: Identify a public artifact (article, blog post, video, etc.) released between June 1, 2023, and December 4, 2023, highlighting an aspect of AI misuse in a regulated domain or impacting a legally recognized protected class. The artifact must be supported by data evidence (research, dataset, survey results, etc.).
- **Deliverables**: Provide the artifact's title, release date, link, application/scenario/domain of misuse, the impacted regulated domain/protected class, and a link to the evidence.

The iTutorGroup case, settled with the Equal Employment Opportunity Commission (EEOC) in August 2023, highlights critical issues in AI ethics, particularly in the context of AI-driven hiring processes and age discrimination. Below are the key findings from multiple sources:

1. **Nature of the Settlement**:
   - iTutorGroup agreed to pay $365,000 to settle an AI lawsuit with the EEOC, related to allegations of age discrimination in its hiring process【14†source】【20†source】【26†source】.
   - The lawsuit claimed iTutorGroup's AI-powered hiring software illegally screened out older job applicants, specifically women over 55 and men over 60【14†source】【20†source】.

2. **Background and Implications**:
   - This was the first lawsuit by the EEOC involving a company's use of AI for employment decisions【26†source】.
   - The EEOC, enforcing workplace bias laws, had warned about the potential misuse of AI in such contexts and launched an initiative in 2021 to ensure AI software complies with anti-discrimination laws【26†source】.
   - iTutorGroup, while denying wrongdoing, agreed to several corrective measures, including adopting new anti-discrimination policies and conducting anti-discrimination trainings【14†source】.

3. **Broader Context in AI and Employment**:
   - A significant portion (around 79% to 85%) of U.S. employers reportedly use AI in various aspects of employment, including applicant screening, HR chatbots, performance reviews, and promotion recommendations【14†source】【26†source】.
   - This widespread use of AI in employment raises concerns about potential biases being embedded in AI systems, even unintentionally【26†source】.
   - The iTutorGroup case represents a growing trend where lawsuits may increasingly challenge employers for discriminatory practices through AI software【26†source】.

This case serves as a crucial reminder of the ethical responsibilities in implementing AI technologies, especially in sensitive areas like employment. It underscores the need for rigorous checks and balances to prevent and address potential biases and discrimination in AI systems.

---

# Question 2
Bias Summary: iTutorGroup AI Hiring Discrimination Case

The iTutorGroup case with the Equal Employment Opportunity Commission (EEOC) vividly illustrates the concept of **algorithmic bias** in AI systems, specifically in the domain of **AI-driven hiring processes**. Algorithmic bias occurs when an AI system reflects and perpetuates existing prejudices, leading to unfair outcomes for certain groups of individuals. In this instance, iTutorGroup’s AI-based application review software was programmed to automatically reject applications from female applicants aged 55 or older and male applicants aged 60 or older. This bias explicitly manifests as **age discrimination**, contravening the principles of **fairness and equality** in employment as outlined by the **Age Discrimination Act**.

The bias identified in this case is a quintessential example of **explicit bias**, where the AI algorithm was deliberately designed to discriminate based on age. This contrasts with **implicit bias**, which often arises from skewed training data or flawed algorithm design without explicit intent to discriminate. The discriminatory practice in iTutorGroup's AI system underscores the importance of **ethical AI design** and **algorithmic accountability**. It highlights the imperative for AI systems, particularly those used in regulated domains like employment, to be developed and audited in alignment with **ethical guidelines and legal standards**. The case also exemplifies the **societal impacts** of AI technologies, where biased AI hiring tools can significantly affect the lives and opportunities of protected classes. Furthermore, it emphasizes the role of regulatory bodies like the EEOC in ensuring AI technologies are used in ways that uphold **anti-discrimination laws** and foster an environment of **inclusive and equitable employment practices**.


## Overview of Algorithmic Bias in AI-Driven Hiring
The iTutorGroup case with the EEOC serves as a seminal example of **algorithmic bias** in AI systems, particularly within **AI-driven hiring processes**. Algorithmic bias occurs when an AI system perpetuates or even amplifies pre-existing prejudices, resulting in unfair treatment of certain groups.

### Nature of Bias in iTutorGroup's AI System
- **Explicit Age Discrimination**: iTutorGroup's AI system was programmed to reject applications from female applicants aged 55 or older and male applicants aged 60 or older.
- **Contravention of Fairness Principles**: This practice directly contravenes principles of **fairness and equality** in employment, violating the **Age Discrimination Act**.

### **Explicit vs. Implicit Bias**
- **Explicit Bias**: The iTutorGroup case is an example of explicit bias, where age-based discrimination was intentionally programmed into the algorithm.
- **Implicit Bias Contrast**: This contrasts with implicit bias, often stemming from skewed training data or unintentional flaws in algorithm design.

## Ethical AI Design and Algorithmic Accountability
- **Importance of Ethical Design**: iTutorGroup's discriminatory AI practice underscores the need for **ethical AI design**.
- **Algorithmic Accountability**: It raises critical questions about **algorithmic accountability**, particularly in how AI systems are developed and audited.
- **Adherence to Ethical Guidelines and Legal Standards**: The case highlights the necessity for AI systems, especially in regulated fields like employment, to align with ethical guidelines and legal standards.

## Societal Impacts of Biased AI Hiring Tools
- **Effect on Protected Classes**: Biased AI hiring tools can have significant repercussions on the lives and opportunities of protected classes.
- **Role of Regulatory Bodies**: The role of bodies like the EEOC in upholding **anti-discrimination laws** and promoting **inclusive and equitable employment practices** is crucial.

## Investigative Deep-Dive into the iTutorGroup Case
- **Prevalence of AI in Hiring**: The widespread use of AI in hiring processes raises concerns about systemic biases and the need for robust ethical frameworks.
- **Regulatory Responses and Legal Implications**: The EEOC’s response to the iTutorGroup case sets a precedent for how such issues might be handled legally and regulatorily in the future.
- **Need for Transparency in AI Algorithms**: This case emphasizes the importance of transparency in AI algorithms to prevent discriminatory practices.
- **Broader Industry Implications**: The settlement may have far-reaching implications for the HR tech industry, encouraging more rigorous standards for AI-driven hiring tools.

## Conclusion
The iTutorGroup case is a bellwether for the ethical challenges posed by AI in hiring. It serves as a stark reminder of the potential societal consequences of unchecked algorithmic bias and the importance of ethical, accountable AI systems in regulated domains.


---

# Question 3

## Metrics Overview
This section details the quantifiable metrics derived from the iTutorGroup AI discrimination case. Each metric is categorized and explained with its relevance to the case and broader AI ethics considerations.

### 1. **Privileged/Unprivileged Groups**
   - **Privileged Group**: Applicants under the age of 55 (for females) and 60 (for males).
   - **Unprivileged Group**: Female applicants aged 55 or older and male applicants aged 60 or older.
   - **Importance**: This metric highlights the age-based discrimination inherent in the AI system, segregating applicants into privileged (younger) and unprivileged (older) groups based on age.

### 2. **Data Bias Sources**
   - **Source of Bias**: Programmed discrimination in the application review algorithm.
   - **Importance**: The bias was intentionally embedded in the algorithm, showcasing a clear source of data bias where the AI system was designed to discriminate based on age.

### 3. **Sampling Bias Sources**
   - **Potential Bias in Data Collection**: The data (applicant information) fed into the AI system likely did not represent a diverse age range equally.
   - **Importance**: This suggests a sampling bias where older applicants were systematically disadvantaged, impacting the fairness of the hiring process.

### 4. **Sampling Methods**
   - **Method Used**: Automated AI screening of applications.
   - **Importance**: The method of using AI for initial screening can introduce bias if the algorithm is not designed to handle diversity in applicant demographics fairly.

### 5. **Correlations**
   - **Correlated Factors**: Age and application rejection.
   - **Importance**: A strong correlation was found between applicant age and the likelihood of being rejected, indicating discriminatory practices.

### 6. **Outcome Measures**
   - **Measure**: Rejection rates based on age.
   - **Importance**: This outcome measure is crucial for understanding the direct impact of the biased AI system on older applicants.

### 7. **Bias & Fairness Metrics**
   - **Metric**: Disparity in interview offer rates between younger and older applicants.
   - **Importance**: This metric is essential for quantifying the extent of unfair treatment and assessing the AI system's compliance with fairness norms.

### 8. **Legal and Regulatory Compliance**
   - **Compliance Failure**: Violation of the Age Discrimination Act.
   - **Importance**: This metric is critical for evaluating the legal implications of the AI system's bias, particularly in the context of regulated domains like employment.

## Conclusion
The iTutorGroup case serves as a stark reminder of the potential for AI systems to perpetuate biases, emphasizing the need for thorough and ethical design, implementation, and monitoring of AI technologies, especially in regulated domains like employment.

---

To address the final section on Mitigation Method Design, I will provide a comprehensive and informative set of Markdown notes. These notes will identify a specific issue related to the quantifiable metrics from the iTutorGroup case and propose a method to mitigate this bias, using pseudo-code to illustrate the approach:

# Question 4

## Identified Issue: Age-Based Discrimination
- **Quantifiable Metric**: The disparity in interview offer rates between younger and older applicants, as evidenced by iTutorGroup’s AI system rejecting female applicants aged 55 or older and male applicants aged 60 or older.

## Proposed Mitigation Method
The proposed method focuses on **fairness in algorithmic decision-making**, aiming to reduce age-based discrimination in AI-driven hiring processes. The concept aligns with lectures on **ethical AI design** and **bias mitigation** techniques.

### **Method Overview**
- **Objective**: To redesign the AI hiring algorithm to ensure fairness across age groups.
- **Approach**: Implement a fairness-aware machine learning model that incorporates anti-discrimination constraints.

### **Pseudo-Code for Fairness-Aware Model**

```python
# Pseudo-Code for Fairness-Aware Hiring Algorithm

# Import necessary machine learning libraries
from fairness_aware_ml_library import FairnessAwareClassifier

# Load applicant data
applicant_data = load_data("applicant_dataset.csv")

# Define sensitive attribute (age)
sensitive_attribute = "age"

# Initialize the Fairness-Aware Classifier
fair_classifier = FairnessAwareClassifier(sensitive_attribute=sensitive_attribute)

# Train the model with fairness constraints
fair_classifier.fit(applicant_data)

# Use the trained model for predicting interview offers
predictions = fair_classifier.predict(applicant_data)

# Ensure fairness in predictions
ensure_fairness(predictions, sensitive_attribute)
```

### **Data Inputs and Outputs**
- **Inputs**: Applicant data including qualifications, experience, and age.
- **Outputs**: Fair and unbiased interview offer decisions.

### **Anticipated Change in Outcomes**
- **Reduction in Age Bias**: The algorithm is expected to significantly reduce age-based discrimination in hiring decisions.
- **Compliance with Legal Standards**: The new model aims to align with anti-discrimination laws like the Age Discrimination Act.
- **Enhanced Diversity in Hiring**: By mitigating age bias, the organization is likely to see a more diverse range of applicants advancing in the hiring process.

## Conclusion
The redesign of the AI hiring system using a fairness-aware approach represents a crucial step towards ethical AI practices in employment. By addressing specific quantifiable metrics of bias, this method aims to create a more inclusive and equitable hiring process, upholding both ethical and legal standards in AI-driven decision-making.

This comprehensive set of notes identifies a critical issue related to age discrimination in AI-driven hiring and proposes a mitigation method based on fairness-aware machine learning techniques. The approach aligns with concepts of ethical AI design and bias mitigation, aiming to create a more equitable hiring process.