# **Sensor Failure Detection Model**

**This repository contains an end to end machine learningpipeline built to predict industrial machine failures based on continuous sensor telemetry.**

**Exploratory Data Analysis**
   After performing comprehensive Univariate, Bivariate and Multivariate analysis, several critical operational and structural characteristics were uncovered:
   
   **Univariate Data Analysis**
   
   **Air Temperature:-** Symmetric, Bimodal,Two primary operation zones [peaked at 298K and 302K], High overall variance.

   ![Process Temperature KDE Plot](process_temp_kde.png)
   
   **Process Temperature:-** Almost symmetric ( with a slight [right skew]) , Mutimodal [with prominent peaks around 309K and 310.5K], High overall variance.
   
   ![Process Temperature KDE](process_temp_kde.png)

   **Torque[Nm]:-** Symmetric,Unimodal(Bell Shaped), Moderate variance and  centered around ~40 Nm across a broad operational range (~0 to 80 Nm).

   ![Torque KDE Plot](torque_kde.png)

   **Tool Wear(min):-** Symmetric,Uniformly distributed (flat plateau across ~0 to 240 min), showing an even spread of tool age throughout the dataset.

   ![Tool Wear KDE Plot](tool_wear_kde.png)

   **Rotational Speed(rpm):-** Unimodal , Right skewed distribution with a prominent peak around ~1500 rpm; most operational data stays tightly clustered (low variance    baseline) with a long tail extending toward 2800+ rpm.

   ![Rotational Speed KDE Plot](rotational_speed_kde.png)

   **Bivariate Data Analysis**

   **Air Temperature VS Process Temperature:-** Strong Positive Linear Correlation (~0.85+ Pearson correlation). The overall trend shows a direct proportional             relationship with noticeable clustered operational bands/steps.

   ![Air VS Process Temperature Scatter Plot](air_process_temp_scatter.png)

   **Rotational Speed VS Torque:-** Strong Negative Non-Linear (Curvilinear / Inverse) Relationship ($\sim -0.87$ Pearson correlation). Reflects underlying mechanical     power constraints ($\text{Power} \propto \text{Torque} \times \text{RPM}$).

   ![Rotational Speed VS Torque Scatter Plot](rotational_speed_torque_scatter.png)

   **Machine Type Variant VS Operating Sensors (`sns.barplot`):-** Across Low (L), Medium (M), and High (H) quality variants, baseline mean values for Torque (~40 Nm),    Rotational Speed (~1530 rpm), and Tool Wear remain constant. This confirms that higher failure rates in Type 'L' machines stem from lower structural tolerance          limits rather than harsher operational usage.

   | Failure Mode | Key Feature | Non-Failure (`0`) Profile | Failure (`1`) Profile | Technical Takeaway |
   | :--- | :--- | :--- | :--- | :--- |
   | **TWF** (Tool Wear) | Tool Wear [min] | Symmetric, centered at low wear | Right-skewed, concentrated at high wear | Failures occur predominantly on aged              components (>200 min). |
   | **PWF** (Power) | Torque [Nm] | Symmetric, normal baseline operating range | Strongly left-skewed at peak high torque | High torque overloads the electrical/         mechanical power boundary. |
   | **HDF** (Heat Dissipation) | Process Temp [K] | Slightly right-skewed at lower operational temp | Elevated right-skewed profile at high temp | Overheating            significantly reduces thermal dissipation capabilities. |

   Cross-tabulation analysis (`pd.crosstab`) reveals that individual failure modes are largely independent of machine quality variants (`L`, `M`, `H`), though failure     rates show minor concentrations in lower-tier models.
   

   **Multivariate Analysis**

   
   **Pearson Correlation Matrix:-** From the Pearson correlation heatmap, it can be inferred that air and process temperatures have a strong positive linear               relationship, whereas torque and rotational speed have a strong negative linear correlation. Additionally, failure modes like HDF and PWF show moderate                 positive linear correlations with machine failure. However, from the sensor vs. failure mode heatmap, no significant linear correlation is observed  between raw        sensor readings and individual failure types.

   **Engineered Feature Correlations Across Failure Modes Heatmap** 
   Correlation Analysis: Pearson correlation revealed low linear dependency across failure modes ($r < 0.3$), indicating that machine breakdowns follow non-linear,        threshold-driven mechanics rather than linear trends.
   Model Selection Justification: This lack of linear correlation confirms that standard linear classifiers are insufficient and strongly justifies deploying non-         linear, tree-based models (e.g., Random Forest, XGBoost).

   ![Feature Correlations Across Failure Modes Heatmap](correlation_heatmap.png)

   **Pair Plot Insights**  
   
   **TWF (Tool Wear Failure):** Tool Wear Failure is primarily time- and wear-accumulative ($>200\text{ min}$). It is driven by mechanical wear rather than                instantaneous torque spikes.

  **OSF (Overstrain Failure):** Overstrain is a multi-variable failure. It occurs when a heavily worn component experiences high mechanical force ($\text{Tool Wear}      \times   \text{Torque}$). Because torque and rotational speed are inversely proportional, low RPM zones correspond to peak torque risk.

  **PWF (Power Failure):** Power Failure is dictated by the continuous power product ($\text{Power} \propto \text{Torque} \times \text{RPM}$). Failures are triggered     when operation breaches upper power limits ($>9000\text{ W}$) or drops below minimum thresholds ($<3500\text{ W}$), rather than single-sensor extremes.

  **Dedicated Scatter Plot Analysis (HDF):**
  **HDF (Heat Dissipation Failure):** A focused scatter plot of `Process temperature` vs. `Air temperature` isolates thermal failure points. HDF triggers almost          exclusively when the temperature differential ($\Delta T = T_{\text{process}} - T_{\text{air}}$) falls below critical dissipation limits under elevated ambient         temperatures.
