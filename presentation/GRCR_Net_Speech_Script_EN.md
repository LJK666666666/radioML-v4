# GRCR-Net Academic Presentation Speech Script

## Opening (1 minute)

Distinguished committee members and colleagues, thank you for your time!

I am Junkai Li from the College of Information Engineering at Zhejiang University of Technology. Today, I am honored to present our research work: **GRCR-Net: A Complex Residual Network with GPR Denoising and Rotational Augmentation for Automatic Modulation Classification**.

This research addresses a core challenge in wireless communications—automatic modulation classification in complex electromagnetic environments, particularly under low signal-to-noise ratio conditions. Our method achieves **65.38% classification accuracy** on the public dataset, surpassing existing state-of-the-art approaches.

## Section 1: Background and Challenges (3 minutes)

### The Critical Importance of AMC

Automatic Modulation Classification (AMC) serves as a cornerstone technology in modern wireless communication systems, playing irreplaceable roles across multiple critical domains:

1. **Cognitive Radio**: Requires dynamic identification and adaptation to different modulation schemes for intelligent spectrum management
2. **Spectrum Monitoring**: Regulatory authorities need to identify unauthorized signal transmissions
3. **Military Communications**: Signal reconnaissance and identification in electronic warfare
4. **5G/6G Networks**: Intelligent signal processing and adaptive communications

### The Core Challenge

However, existing methods face severe challenges:

**Limitations of Traditional Approaches:**
- Likelihood-based methods, while theoretically optimal, suffer from extremely high computational complexity and require precise channel parameters
- Feature-based methods rely on expert-designed features with limited generalization capability
- Deep learning methods, despite excellent performance, experience dramatic accuracy degradation under low SNR conditions

**Critical Technical Issues:**
1. **Severe Noise Interference**: In low SNR conditions, noise severely affects signal quality, causing significant classification accuracy drops
2. **Phase Information Loss**: Traditional real-valued networks cannot effectively process the complex nature of I/Q signals, leading to phase information loss
3. **Data Imbalance**: Uneven data distribution across different modulation types and SNR conditions affects model generalization

As shown in the performance curve, traditional methods perform far worse than our GRCR-Net approach in low SNR environments. This is precisely the core problem we aim to solve.

## Section 2: GRCR-Net Innovation Overview (2 minutes)

Facing these challenges, we propose GRCR-Net, a breakthrough method integrating three core innovations:

### Three Core Innovations

1. **Adaptive GPR Denoising**: An intelligent denoising algorithm based on Gaussian Process Regression that adaptively adjusts denoising strength according to signal-to-noise ratio
2. **Rotational Data Augmentation**: Exploits the geometric symmetry of modulation signal constellations through rotational transformations to expand training data
3. **Hybrid ComplexCNN-ResNet Architecture**: Fuses the advantages of complex convolutional neural networks and residual networks, implementing deep learning in the complex domain

### Synergistic Effects

These three technologies are not simply additive but produce significant synergistic effects:
- GPR denoising contributes 5.86 percentage points improvement
- Rotational augmentation contributes 3.78 percentage points improvement  
- Hybrid architecture achieves 8.27 percentage points improvement over single methods
- **Final achievement of 65.38% classification accuracy**, surpassing existing SOTA methods

## Section 3: Core Technical Details (6 minutes)

### Innovation 1: Adaptive GPR Denoising Algorithm

**Theoretical Foundation**

Our GPR denoising algorithm is built on rigorous mathematical foundations with three core assumptions:

1. **Additive White Gaussian Noise (AWGN)**: The received signal follows the model r[n] = s[n] + w[n]
2. **Independent Gaussian Distribution**: Each noise sample is independent with I/Q components following Gaussian distribution
3. **Gaussian Process Modeling**: The entire signal can be modeled as a Gaussian Process

**Complete Mathematical Derivation**

The key breakthrough is our complete derivation of noise standard deviation estimation through a four-step process:

**Step 1 - I/Q Power Calculation:**
```
Pᵣ = (1/M) Σ(rᵢ[k]² + rQ[k]²)
```

**Step 2 - SNR Power Ratio:**
```
SNR_linear = 10^(SNR_dB/10)
```

**Step 3 - Noise Power Calculation:**
```
Pᵨ = Pᵣ / (10^(SNR_dB/10) + 1)
```

**Step 4 - Final Noise Standard Deviation:**
```
σₙ = √(Pᵣ / (2(10^(SNR_dB/10) + 1)))
```

**Critical Insight**

At SNR = -20dB, noise dominates with 99% of total power. Our derivation provides the exact mathematical relationship to estimate this noise, which becomes the noise variance parameter α in our GPR model, enabling theoretically grounded denoising.

**Adaptive Length-Scale Strategy**

We also implement an SNR-based adaptive length-scale strategy that adjusts denoising strength according to signal conditions, ensuring optimal performance across all SNR ranges.

**Significance of Mathematical Rigor**

This complete mathematical derivation is crucial because it provides:
- **Theoretical Foundation**: Unlike heuristic approaches, our method has solid mathematical backing
- **Parameter Justification**: Every parameter in our GPR model has clear physical meaning
- **Reproducible Results**: The mathematical framework ensures consistent performance across different scenarios
- **Scientific Contribution**: Bridges the gap between signal processing theory and deep learning practice

### Innovation 2: Geometric Symmetry-Based Rotational Data Augmentation

**Theoretical Foundation**

Digital modulation signals possess natural geometric symmetry. Observing constellation diagrams reveals that PSK and QAM modulations exhibit regular symmetric distributions in the complex plane.

**Mathematical Implementation**

We implement complex plane rotation through rotation matrices:
```
[s'ᵢ[n]]   [cos θ  -sin θ] [sᵢ[n]]
[s'Q[n]] = [sin θ   cos θ] [sQ[n]]
```

**Augmentation Strategy**

- Apply 90°, 180°, 270° rotations to training data
- Expand training dataset to 4× original size
- Apply only to modulation types with rotational symmetry

**Performance Improvement**

This strategy significantly improves model robustness to phase offset, particularly for QAM16 and QAM64 modulations, with improvements exceeding 20 percentage points.

### Innovation 3: Hybrid ComplexCNN-ResNet Architecture

**Design Philosophy**

Our hybrid architecture fuses advantages of two network types:
- **ComplexCNN**: Naturally suited for I/Q complex signal processing, preserving phase information
- **ResNet**: Solves gradient vanishing problems in deep networks through residual connections

**Key Technical Components**

1. **Complex Convolutional Layers**: Direct convolution operations in complex domain
2. **ModReLU Activation Function**: Preserves complex geometric structure
3. **Complex Residual Blocks**: Implements residual learning in complex domain
4. **Complex Batch Normalization**: Stabilizes training process

**Architectural Advantages**

- Entire network operates in complex domain until final classification layer conversion to real domain
- Lightweight design with moderate parameter count
- Stable training with fast convergence

## Section 4: Experimental Results and Analysis (4 minutes)

### Experimental Setup

We conducted comprehensive evaluation on the standard RML2016.10a dataset:
- 11 modulation types, 22,000 samples each
- SNR range from -20dB to +18dB
- Data split ratio: 72% training, 8% validation, 20% testing

The experimental environment uses high-performance workstations to ensure result reliability and reproducibility.

### Baseline Comparison Results

We conducted comprehensive comparisons with multiple mainstream methods:
- Traditional approaches: FCNN, CNN1D, CNN2D
- Advanced architectures: Transformer, ResNet, ComplexCNN

Results demonstrate that our GRCR-Net achieves **65.38% accuracy**, representing an **8.27 percentage point improvement** over the best baseline method ComplexCNN.

### State-of-the-Art Comparison

More importantly, we compared with the latest published methods:
- **LDCVNN (2025)**: 62.41% - Dual-branch complex network
- **ULCNN (2022)**: 62.47% - Ultra-lightweight CNN
- **AMC-NET (2023)**: 62.51% - Frequency-domain denoising
- **HFECNET-CA (2023)**: 63.92% - Attention mechanism
- **AbFTNet (2024)**: 64.59% - Previous SOTA with multimodal fusion

Our **GRCR-Net achieves 65.38%**, establishing a new state-of-the-art benchmark with clear performance leadership across all model sizes from lightweight to normal architectures.

### Ablation Study Analysis

Ablation experiments reveal component contributions:
- Baseline hybrid architecture: 56.94%
- Adding GPR denoising: 62.80% (+5.86%)
- Adding rotational augmentation: 60.72% (+3.78%)
- Complete GRCR-Net: 65.38% (+8.44%)

This proves the effectiveness and synergistic effects of each technical component.

### SNR Performance Analysis

Most impressively, our performance across different SNR conditions:
- **Low SNR (-20dB to -8dB)**: Average improvement of 7.25 percentage points
- **Medium SNR (-6dB to 4dB)**: Average improvement of 5.12 percentage points
- **High SNR (6dB to 18dB)**: Average improvement of 5.07 percentage points

This demonstrates stable performance advantages of our method across all signal-to-noise ratio conditions.

## Section 5: Technical Contributions and Impact (2 minutes)

### Major Technical Contributions

Our research makes groundbreaking contributions in three key areas:

1. **Theoretically Grounded GPR Denoising**:
   - Complete mathematical derivation from AWGN assumptions to noise estimation
   - First rigorous theoretical foundation for GPR-based AMC denoising
   - Adaptive length-scale strategy based on SNR conditions
   - Provides new paradigm for signal processing in complex electromagnetic environments

2. **Geometric Symmetry Data Augmentation**:
   - Exploits inherent rotational symmetry of modulation constellations
   - Mathematically rigorous rotation matrix implementation
   - Effective solution for data-scarce scenarios with 4× data expansion
   - Significant robustness improvement against phase offset

3. **Hybrid Complex-Domain Architecture Innovation**:
   - First successful fusion of ComplexCNN and ResNet advantages
   - Deep residual learning implemented entirely in complex domain
   - Preserves phase information while solving gradient vanishing
   - New architectural paradigm for I/Q signal processing

### Comparative Advantages over Existing Methods

Our comprehensive comparison with state-of-the-art methods shows clear advantages:

**Performance Leadership:**
- **65.38% accuracy** - New SOTA achievement
- **+0.79%** improvement over AbFTNet (2024, Previous SOTA)
- **+2.87%** improvement over AMC-NET (2023, ICASSP)
- **+2.91%** improvement over ULCNN (2022)

**Technical Innovation:**
- Organic fusion of three core technologies
- First theoretically grounded GPR denoising for AMC
- Novel hybrid ComplexCNN-ResNet architecture

**Practical Robustness:**
- Stable performance across all SNR conditions
- Consistent improvements over existing methods
- Robust performance in complex electromagnetic environments

**Model Efficiency:**
- Normal model size with superior performance
- Balanced approach between lightweight and heavy models
- Scalable components for independent application

### Practical Application Value

Our method has broad application prospects:
- Intelligent spectrum sensing in cognitive radio
- Signal reconnaissance and identification in electronic warfare
- Spectrum monitoring in communication regulation
- Intelligent signal processing in 5G/6G networks

## Section 6: Conclusion and Future Work (2 minutes)

### Research Summary

GRCR-Net represents an important breakthrough in the automatic modulation classification field:

**Core Achievements:**
- Achieved 65.38% classification accuracy, surpassing existing SOTA methods
- Exceptional performance in low SNR environments, solving key challenges in practical applications
- Proposed three core technical innovations with important theoretical and practical value

**Technical Breakthroughs:**
- Adaptive GPR denoising algorithm
- Geometric symmetry data augmentation strategy
- Hybrid ComplexCNN-ResNet architecture

**Impact Significance:**
- Provides effective solutions for complex electromagnetic environments
- Advances cognitive radio technology development
- Offers new research insights for the signal processing field

### Future Research Directions

We plan to continue in-depth research in the following directions:

1. **Algorithm Optimization and Extension**: Explore performance in more complex channel environments, research real-time processing optimization strategies
2. **Technology Fusion and Innovation**: Combine with emerging architectures like Transformers, explore multimodal signal fusion
3. **Practical Deployment and Applications**: Hardware acceleration optimization, large-scale real-environment validation

## Conclusion (30 seconds)

In summary, GRCR-Net achieves breakthrough progress in automatic modulation classification tasks through three core technical innovations. Our method not only has important academic value but also demonstrates tremendous potential in practical applications.

We have open-sourced the complete code and data, hoping to promote development of the entire field.

Thank you for your attention! I welcome questions and discussions from all committee members.

---

**Presentation Time Allocation:**
- Opening: 1 minute
- Background: 3 minutes
- Method Overview: 2 minutes
- Technical Details: 6 minutes (emphasize GPR mathematical derivation)
- Experimental Results: 4 minutes (highlight SOTA comparison)
- Contributions Impact: 2 minutes
- Conclusion Future: 2 minutes
- **Total: 20 minutes**

**Presentation Key Points:**
1. Maintain passionate and confident tone
2. Emphasize key data points (65.38% SOTA, +0.79% improvement)
3. Highlight technical innovations (especially GPR mathematical derivation)
4. Maintain eye contact with audience
5. Use appropriate gestures to assist expression
6. **Special emphasis**: Explain the complete GPR derivation process clearly
7. **Key message**: Theoretical rigor combined with practical excellence
